from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_fragment_attack as attack


def _fixture() -> np.ndarray:
    yy, xx = np.mgrid[:96, :128]
    return (
        np.stack(
            [
                40.0 + 0.8 * xx,
                50.0 + 0.7 * yy,
                30.0 + 0.4 * xx + 0.3 * yy,
            ],
            axis=2,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


def test_affine_combo_preserves_geometry_and_is_deterministic() -> None:
    source = _fixture()

    first = attack.affine_combo(source, rotation_degrees=0.2, zoom=0.004)
    second = attack.affine_combo(source, rotation_degrees=0.2, zoom=0.004)

    assert first.shape == source.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_bounded_smooth_warp_preserves_geometry_and_is_deterministic() -> None:
    source = _fixture()

    first = attack.bounded_smooth_warp(source, max_displacement=1.8, sigma=8.0, seed=17)
    second = attack.bounded_smooth_warp(source, max_displacement=1.8, sigma=8.0, seed=17)

    assert first.shape == source.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, source)


def test_zero_bounded_warp_is_pixel_identical() -> None:
    source = _fixture()

    result = attack.bounded_smooth_warp(source, max_displacement=0.0, sigma=8.0, seed=17)

    assert np.array_equal(result, source)


def test_color_nudge_is_bounded_and_changes_pixels() -> None:
    source = _fixture()

    result = attack.color_nudge(
        source,
        brightness=0.004,
        contrast=0.006,
        saturation=-0.005,
        hue_degrees=0.15,
    )

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, source)


def test_jpeg_chain_rejects_bad_quality() -> None:
    with pytest.raises(ValueError, match="quality"):
        attack.jpeg_chain(_fixture(), (94, 101))


def test_candidate_batch_has_control_target_and_sham() -> None:
    candidates = attack.build_candidates(_fixture())

    assert np.array_equal(candidates["control"], _fixture())
    assert "fragment-balanced" in candidates
    assert "fragment-strong" in candidates
    assert "sham-strong-rms" in candidates
    assert "bounded-fragment-balanced" in candidates
    assert "bounded-fragment-strong" in candidates
    assert "sham-bounded-strong-rms" in candidates
    assert all(pixels.shape == _fixture().shape for pixels in candidates.values())

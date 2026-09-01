"""Tests for the research adaptive periodic-carrier suppressor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_adaptive_carrier_suppress as suppressor  # pyright: ignore[reportMissingImports]


def repeated_template(height: int, width: int, amplitude: float) -> NDArray[Any]:
    """Return a synthetic uint8 image carrying the bundled periodic template."""
    template, _sigma, *_model = suppressor._load_template()
    repeats_y = (height + template.shape[0] - 1) // template.shape[0]
    repeats_x = (width + template.shape[1] - 1) // template.shape[1]
    carrier = np.tile(template, (repeats_y, repeats_x, 1))[:height, :width]
    return np.clip(np.rint(128.0 + amplitude * carrier), 0, 255).astype(np.uint8)


def test_apply_template_handles_nondivisible_geometry() -> None:
    template, _sigma, *_model = suppressor._load_template()
    pixels = np.full((65, 67, 3), 128, dtype=np.uint8)

    candidate = suppressor.apply_template(pixels, template, amplitude=8.0)

    assert candidate.shape == pixels.shape
    assert candidate.dtype == np.uint8
    assert np.any(candidate != pixels)


def test_find_minimum_amplitude_reaches_target() -> None:
    template, sigma, *_model = suppressor._load_template()
    pixels = repeated_template(128, 130, 80.0)

    amplitude, candidate, score = suppressor.find_minimum_amplitude(
        pixels,
        template,
        sigma,
        target_score=-0.25,
        maximum_amplitude=160.0,
        iterations=10,
    )

    assert 0.0 < amplitude <= 160.0
    assert score <= -0.25
    assert suppressor.carrier_score(candidate, template, sigma) == pytest.approx(score)


def test_find_minimum_amplitude_rejects_unreachable_target() -> None:
    template, sigma, *_model = suppressor._load_template()
    pixels = repeated_template(128, 128, 80.0)

    with pytest.raises(ValueError, match="maximum amplitude"):
        suppressor.find_minimum_amplitude(
            pixels,
            template,
            sigma,
            target_score=-0.25,
            maximum_amplitude=1.0,
            iterations=8,
        )


def test_suppress_carrier_refuses_local_negative() -> None:
    pixels = np.full((1000, 1000, 3), 128, dtype=np.uint8)

    with pytest.raises(ValueError, match="not detected"):
        suppressor.suppress_carrier(pixels)

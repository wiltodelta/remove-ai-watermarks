from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_v3_carrier_subtract as subtract
from synthid_v3_codebook_probe import V3CarrierModel


def _model() -> V3CarrierModel:
    height = width = 64
    return V3CarrierModel(
        height=height,
        width=width,
        rows=np.asarray([7, 7, 7], dtype=np.int32),
        columns=np.asarray([5, 5, 5], dtype=np.int32),
        channels=np.asarray([0, 1, 2], dtype=np.int8),
        phases=np.asarray([0.4, 0.4, 0.4]),
        weights=np.full(3, 1.0 / 3.0),
        expected_magnitudes=np.full(3, 20.0 * height * width / 2.0),
    )


def _carrier(model: V3CarrierModel) -> np.ndarray:
    yy, xx = np.mgrid[: model.height, : model.width]
    values = 100.0 + 20.0 * np.cos(2.0 * np.pi * (7.0 * yy / model.height + 5.0 * xx / model.width) + 0.4)
    return np.repeat(values[:, :, None], 3, axis=2).round().astype(np.uint8)


def test_subtract_carrier_preserves_shape_and_removes_known_component() -> None:
    model = _model()
    source = _carrier(model)

    result = subtract.subtract_carrier(source, model, strength=1.0)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert np.std(result.astype(np.float64)) < 1.0


def test_zero_strength_is_pixel_identical() -> None:
    model = _model()
    source = _carrier(model)

    result = subtract.subtract_carrier(source, model, strength=0.0)

    assert np.array_equal(result, source)


def test_rejects_wrong_geometry() -> None:
    model = _model()

    with pytest.raises(ValueError, match="does not match"):
        subtract.subtract_carrier(np.zeros((32, 32, 3), dtype=np.uint8), model, strength=1.0)


@pytest.mark.parametrize("value", ["1,0.5", "0.5,0.5", "-1,0.5", "x"])
def test_rejects_invalid_strength_sweep(value: str) -> None:
    with pytest.raises(click.BadParameter):
        subtract.parse_strengths(value)

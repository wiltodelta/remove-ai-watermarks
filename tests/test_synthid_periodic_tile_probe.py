"""Tests for the periodic spatial-carrier research probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import synthid_periodic_tile_probe as probe
from synthid_periodic_tile import cyclic_tile_correlations


def _carrier(seed: int, *, tile_height: int = 8, tile_width: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tile = rng.normal(size=(tile_height, tile_width, 3))
    return tile - np.mean(tile, axis=(0, 1), keepdims=True)


def _write_image(path: Path, carrier: np.ndarray, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    repeated = np.tile(carrier, (8, 8, 1))
    yy, xx = np.mgrid[:64, :64]
    background = 110.0 + 0.2 * xx + 0.1 * yy
    pixels = background[:, :, None] + 18.0 * repeated + rng.normal(scale=1.5, size=repeated.shape)
    Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB").save(path)


def test_periodic_model_scores_matching_carrier_and_round_trips(tmp_path: Path) -> None:
    carrier = _carrier(1)
    positives = []
    for index in range(4):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, carrier, seed=index)
        positives.append(path)
    heldout = tmp_path / "heldout.png"
    negative = tmp_path / "negative.png"
    _write_image(heldout, carrier, seed=10)
    _write_image(negative, _carrier(2), seed=11)

    model = probe.discover_model(positives, tile_height=8, tile_width=8)
    matching = probe.score_image(heldout, model)
    mismatching = probe.score_image(negative, model)
    model_path = tmp_path / "model.npz"
    probe.save_model(model_path, model)
    restored = probe.load_model(model_path)

    assert matching.score > 0.9
    assert mismatching.score < 0.5
    assert matching.active_support > 0.0
    assert matching.repeat_count == 64
    assert np.array_equal(restored.template, model.template)
    assert probe.score_image(heldout, restored).score == matching.score


def test_registration_recovers_cyclic_tile_shift(tmp_path: Path) -> None:
    carrier = _carrier(3)
    positives = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, carrier, seed=index)
        positives.append(path)
    source = tmp_path / "source.png"
    shifted = tmp_path / "shifted.png"
    _write_image(source, carrier, seed=20)
    with Image.open(source) as image:
        pixels = np.asarray(image).copy()
    Image.fromarray(np.roll(pixels, shift=(1, 2), axis=(0, 1)), mode="RGB").save(shifted)
    model = probe.discover_model(positives, tile_height=8, tile_width=8)

    fixed = probe.score_image(shifted, model)
    registered = probe.score_image(shifted, model, register=True)

    assert registered.score > fixed.score
    assert registered.score > 0.9
    assert (registered.row_shift, registered.column_shift) == (7, 6)


def test_array_scoring_matches_file_scoring(tmp_path: Path) -> None:
    carrier = _carrier(12)
    positives = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, carrier, seed=index)
        positives.append(path)
    model = probe.discover_model(positives, tile_height=8, tile_width=8)
    with Image.open(positives[0]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)

    file_score = probe.score_image(positives[0], model)
    array_score = probe.score_pixels(pixels, model)

    assert array_score.score == pytest.approx(file_score.score)
    assert array_score.active_support == pytest.approx(file_score.active_support)
    assert array_score.path == "<array>"


def test_fft_correlations_match_explicit_cyclic_shifts() -> None:
    rng = np.random.default_rng(30)
    template = rng.normal(size=(4, 5, 3))
    tile = rng.normal(size=(4, 5, 3))
    explicit = np.asarray(
        [
            [np.sum(template * np.roll(tile, shift=(row, column), axis=(0, 1))) for column in range(5)]
            for row in range(4)
        ]
    )

    correlations = cyclic_tile_correlations(template, tile)

    assert correlations == pytest.approx(explicit)


def test_calibration_is_strictly_above_every_negative(tmp_path: Path) -> None:
    carrier = _carrier(4)
    positives = []
    negatives = []
    for index in range(3):
        positive = tmp_path / f"positive-{index}.png"
        negative = tmp_path / f"negative-{index}.png"
        _write_image(positive, carrier, seed=index)
        _write_image(negative, _carrier(10 + index), seed=20 + index)
        positives.append(positive)
        negatives.append(negative)
    model = probe.discover_model(positives, tile_height=8, tile_width=8)

    threshold = probe.calibrate_threshold(negatives, model)

    assert all(probe.score_image(path, model).score < threshold for path in negatives)
    with pytest.raises(ValueError, match="at least one calibration negative"):
        probe.calibrate_threshold([], model)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_color_space_probe as probe


def _write_image(path: Path, *, phase: float | None, seed: int) -> None:
    height = width = 64
    rng = np.random.default_rng(seed)
    pixels = 100.0 + rng.normal(0.0, 3.0, size=(height, width, 3))
    if phase is not None:
        yy, xx = np.mgrid[:height, :width]
        pixels[:, :, 0] += 12.0 * np.cos(2.0 * np.pi * (7.0 * yy / height + 5.0 * xx / width) + phase)
        pixels[:, :, 1] -= 8.0 * np.cos(2.0 * np.pi * (7.0 * yy / height + 5.0 * xx / width) + phase)
    Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB").save(path)


@pytest.mark.parametrize("color_space", probe.COLOR_SPACES)
def test_color_transforms_are_finite_and_preserve_shape(color_space: str) -> None:
    rgb = np.asarray([[[0.0, 128.0, 255.0], [64.0, 64.0, 64.0]]])

    transformed = probe.transform_color_space(rgb, color_space)

    assert transformed.shape == rgb.shape
    assert np.all(np.isfinite(transformed))


def test_neutral_gray_has_neutral_chroma() -> None:
    gray = np.full((1, 1, 3), 96.0)

    ycbcr = probe.transform_color_space(gray, "ycbcr")
    ycocg = probe.transform_color_space(gray, "ycocg")
    opponent = probe.transform_color_space(gray, "opponent")
    lab = probe.transform_color_space(gray, "lab")
    hsv = probe.transform_color_space(gray, "hsv")

    assert ycbcr[0, 0, 1:] == pytest.approx((128.0, 128.0), abs=1e-4)
    assert ycocg[0, 0, 1:] == pytest.approx((0.0, 0.0), abs=1e-8)
    assert opponent[0, 0, 1:] == pytest.approx((0.0, 0.0), abs=1e-8)
    assert lab[0, 0, 1:] == pytest.approx((0.0, 0.0), abs=0.1)
    assert hsv[0, 0, 1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("color_space", ["rgb", "opponent"])
def test_discovered_model_separates_shared_phase_from_noise(tmp_path: Path, color_space: str) -> None:
    positives: list[Path] = []
    for index in range(4):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, phase=0.4, seed=index)
        positives.append(path)
    heldout = tmp_path / "heldout.png"
    negative = tmp_path / "negative.png"
    _write_image(heldout, phase=0.4, seed=10)
    _write_image(negative, phase=None, seed=11)
    bins = np.asarray([(7, 5, channel) for channel in range(3)], dtype=np.int32)

    model = probe.discover_model(
        positives,
        color_space=color_space,
        candidate_bins=bins,
        peak_count=3,
    )

    assert probe.score_image(heldout, model).evidence_score > probe.score_image(negative, model).evidence_score


def test_model_round_trip_is_pickle_free(tmp_path: Path) -> None:
    positives: list[Path] = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, phase=0.4, seed=index)
        positives.append(path)
    bins = np.asarray([(7, 5, channel) for channel in range(3)], dtype=np.int32)
    model = probe.discover_model(positives, color_space="ycocg", candidate_bins=bins, peak_count=3)
    artifact = tmp_path / "model.npz"

    probe.save_model(artifact, model)
    loaded = probe.load_model(artifact)

    assert loaded.color_space == "ycocg"
    assert np.array_equal(loaded.channels, model.channels)
    assert np.isclose(np.sum(loaded.weights), 1.0)


def test_channel_evidence_adds_to_total(tmp_path: Path) -> None:
    positives: list[Path] = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, phase=0.4, seed=index)
        positives.append(path)
    bins = np.asarray([(7, 5, channel) for channel in range(3)], dtype=np.int32)
    model = probe.discover_model(positives, color_space="rgb", candidate_bins=bins, peak_count=3)

    score = probe.score_image(positives[0], model)

    assert sum(score.channel_evidence) == pytest.approx(score.evidence_score)
    assert sum(score.selected_peak_counts) == score.peak_count

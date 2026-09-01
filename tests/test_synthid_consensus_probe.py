from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_consensus_probe as probe


def _write_pattern(path: Path, base: tuple[int, int, int], *, sign: float, marked: bool) -> None:
    size = 128
    yy, xx = np.mgrid[:size, :size]
    carrier = 2.0 * np.sin(2.0 * np.pi * (11.0 * yy + 7.0 * xx) / size)
    carrier += 1.5 * np.sin(2.0 * np.pi * (17.0 * yy - 5.0 * xx) / size + 0.4)
    pixels = np.broadcast_to(np.asarray(base, dtype=np.float64), (size, size, 3)).copy()
    if marked:
        pixels += sign * carrier[:, :, None]
    Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB").save(path)


def _reference_groups(tmp_path: Path) -> tuple[list[list[Path]], Path, Path]:
    groups: list[list[Path]] = []
    for group_index, base in enumerate(((40, 50, 60), (170, 180, 190))):
        directory = tmp_path / f"group-{group_index}"
        directory.mkdir()
        paths: list[Path] = []
        for image_index in range(3):
            path = directory / f"marked-{image_index}.png"
            sign = -1.0 if group_index == 1 else 1.0
            _write_pattern(path, base, sign=sign, marked=True)
            paths.append(path)
        groups.append(paths)
    positive = tmp_path / "positive.png"
    negative = tmp_path / "negative.png"
    _write_pattern(positive, (100, 110, 120), sign=-1.0, marked=True)
    _write_pattern(negative, (100, 110, 120), sign=1.0, marked=False)
    return groups, positive, negative


def test_discovers_polarity_invariant_carrier(tmp_path: Path) -> None:
    groups, positive, negative = _reference_groups(tmp_path)

    model = probe.discover_model(groups, size=128, peak_count=16, min_radius=3.0)
    positive_score = probe.score_image(positive, model)
    negative_score = probe.score_image(negative, model)

    assert positive_score.score > 0.8
    assert positive_score.active_weight_fraction > 0.5
    assert negative_score.active_weight_fraction < 0.01


def test_model_round_trip_disables_pickle(tmp_path: Path) -> None:
    groups, positive, _ = _reference_groups(tmp_path)
    model = probe.discover_model(groups, size=128, peak_count=8, min_radius=3.0)
    artifact = tmp_path / "model.npz"

    probe.save_model(artifact, model)
    loaded = probe.load_model(artifact)

    assert probe.score_image(positive, loaded).score == pytest.approx(probe.score_image(positive, model).score)
    assert loaded.peaks.dtype == np.int32


def test_requires_independent_groups(tmp_path: Path) -> None:
    groups, _, _ = _reference_groups(tmp_path)

    with pytest.raises(ValueError, match="at least two"):
        probe.discover_model(groups[:1], size=128, peak_count=8)

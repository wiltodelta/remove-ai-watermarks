from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_color_space_probe as probe
import synthid_hybrid_attack as attack


def _write_image(path: Path, *, seed: int) -> None:
    height = width = 64
    rng = np.random.default_rng(seed)
    pixels = 100.0 + rng.normal(0.0, 2.0, size=(height, width, 3))
    yy, xx = np.mgrid[:height, :width]
    wave = np.cos(2.0 * np.pi * (7.0 * yy / height + 5.0 * xx / width) + 0.4)
    pixels[:, :, 0] += 14.0 * wave
    pixels[:, :, 1] -= 10.0 * wave
    Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB").save(path)


def test_hybrid_matrix_preserves_geometry_and_has_controls(tmp_path: Path) -> None:
    positives: list[Path] = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, seed=index)
        positives.append(path)
    source_path = tmp_path / "source.png"
    _write_image(source_path, seed=10)
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bins = np.asarray([(7, 5, channel) for channel in range(3)], dtype=np.int32)
    rgb_model = probe.discover_model(positives, color_space="rgb", candidate_bins=bins, peak_count=3)
    hsv_model = probe.discover_model(positives, color_space="hsv", candidate_bins=bins, peak_count=3)

    candidates = attack.build_candidates(source, rgb_model, hsv_model)

    assert set(candidates) == {
        "projection-075",
        "bounded-100",
        "projection-075-bounded-100",
        "projection-075-bounded-polish",
        "projection-100-elastic-075",
    }
    assert all(candidate.shape == source.shape for candidate in candidates.values())
    assert np.array_equal(candidates["projection-075"], source) is False

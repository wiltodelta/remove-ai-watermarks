from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_color_space_probe as probe
import synthid_ensemble_attack as attack


def _write_image(path: Path, *, phase: float, seed: int) -> None:
    height = width = 64
    rng = np.random.default_rng(seed)
    pixels = 100.0 + rng.normal(0.0, 2.0, size=(height, width, 3))
    yy, xx = np.mgrid[:height, :width]
    wave = np.cos(2.0 * np.pi * (7.0 * yy / height + 5.0 * xx / width) + phase)
    pixels[:, :, 0] += 14.0 * wave
    pixels[:, :, 1] -= 10.0 * wave
    Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB").save(path)


def test_projection_removes_only_positive_phase_component() -> None:
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    channel = np.cos(2.0 * np.pi * (7.0 * yy / height + 5.0 * xx / width) + 0.4)
    before = np.fft.fft2(channel)[7, 5]
    phase = np.angle(before)

    projected = attack._remove_positive_projection(
        channel,
        rows=np.asarray([7]),
        columns=np.asarray([5]),
        phases=np.asarray([phase]),
        strength=1.0,
    )
    after = np.fft.fft2(projected)[7, 5]

    assert np.real(after * np.exp(-1j * phase)) == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(np.imag(np.fft.ifft2(np.fft.fft2(projected))))) < 1e-12


def test_alternating_projection_reduces_rgb_and_sv_evidence(tmp_path: Path) -> None:
    positives: list[Path] = []
    for index in range(3):
        path = tmp_path / f"positive-{index}.png"
        _write_image(path, phase=0.4, seed=index)
        positives.append(path)
    source = tmp_path / "source.png"
    _write_image(source, phase=0.4, seed=10)
    bins = np.asarray([(7, 5, channel) for channel in range(3)], dtype=np.int32)
    rgb_model = probe.discover_model(positives, color_space="rgb", candidate_bins=bins, peak_count=3)
    hsv_model = probe.discover_model(positives, color_space="hsv", candidate_bins=bins, peak_count=3)
    before_rgb = probe.score_image(source, rgb_model)
    before_hsv = probe.score_image(source, hsv_model)

    with Image.open(source) as image:
        source_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    projected = attack.alternating_projection(
        source_pixels,
        rgb_model,
        hsv_model,
        strength=1.0,
        iterations=2,
    )
    output = tmp_path / "projected.png"
    Image.fromarray(projected, mode="RGB").save(output)
    after_rgb = probe.score_image(output, rgb_model)
    after_hsv = probe.score_image(output, hsv_model)

    assert after_rgb.evidence_score < before_rgb.evidence_score
    assert sum(after_hsv.channel_evidence[1:]) < sum(before_hsv.channel_evidence[1:])


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_projection_rejects_out_of_range_strength(strength: float) -> None:
    channel = np.zeros((64, 64))

    with pytest.raises(ValueError, match="between zero and one"):
        attack._remove_positive_projection(
            channel,
            rows=np.asarray([7]),
            columns=np.asarray([5]),
            phases=np.asarray([0.0]),
            strength=strength,
        )

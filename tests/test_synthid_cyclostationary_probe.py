from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_cyclostationary_probe as probe


def _template() -> np.ndarray:
    _y, x = np.indices((16, 16))
    carrier = np.cos(2.0 * np.pi * 4.0 * x / 16.0)
    template = np.stack((carrier, 0.8 * carrier, 0.6 * carrier), axis=2)
    template -= np.mean(template, axis=(0, 1), keepdims=True)
    return template / np.linalg.norm(template)


def test_detects_complex_spectral_coupling() -> None:
    rng = np.random.default_rng(20260814)
    base = rng.normal(0.0, 1.0, (1024, 1024, 3))
    _y, x = np.indices(base.shape[:2])
    modulation = 1.0 + 0.8 * np.cos(2.0 * np.pi * 4.0 * x / 16.0)

    result = probe.score_cyclostationary(
        base * modulation[:, :, None],
        _template(),
        period=16.0,
        harmonic_count=1,
    )

    assert result.selection_contrast > 0.1
    assert result.confirmation_contrast > 0.1
    assert result.joint_contrast > 0.1


def test_rejects_independent_equal_power_noise() -> None:
    rng = np.random.default_rng(20260815)
    noise = rng.normal(0.0, 1.0, (1024, 1024, 3))

    result = probe.score_cyclostationary(
        noise,
        _template(),
        period=16.0,
        harmonic_count=1,
    )

    assert result.joint_contrast < 0.01


def test_does_not_confuse_additive_carrier_with_modulation() -> None:
    rng = np.random.default_rng(20260816)
    noise = rng.normal(0.0, 1.0, (1024, 1024, 3))
    additive = np.tile(_template(), (64, 64, 1)) * 2.0

    result = probe.score_cyclostationary(
        noise + additive,
        _template(),
        period=16.0,
        harmonic_count=1,
    )

    assert result.joint_contrast < 0.01

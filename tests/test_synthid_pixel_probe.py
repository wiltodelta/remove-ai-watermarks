"""Tests for the SynthID pixel-carrier probe.

The probe's logic is validated synthetically: a fixed carrier injected into
uniform fills must correlate strongly, random noise must not, and simulated
removal (dropping the carrier) must collapse the correlation. No real SynthID
fills or GPU are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

# scripts/ is not an installed package; add it to the path for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_pixel_probe as probe

_SHAPE = (64, 64)


def _fixed_carrier(seed: int = 7) -> np.ndarray:
    """A reproducible 2-D carrier pattern (the stand-in for the SynthID signal)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(_SHAPE)


def _fill_with_carrier(base: float, pattern: np.ndarray, noise_seed: int, noise: float = 0.05) -> np.ndarray:
    """A solid fill at brightness ``base`` carrying ``pattern`` plus light noise."""
    rng = np.random.default_rng(noise_seed)
    return base + pattern + noise * rng.standard_normal(_SHAPE)


class TestCarrier:
    def test_flat_image_has_zero_carrier(self):
        # A perfectly uniform fill (std 0, like the synthetic refs) carries nothing.
        flat = np.zeros(_SHAPE)
        assert not np.any(probe.carrier(flat))

    def test_carrier_is_unit_norm(self):
        c = probe.carrier(_fixed_carrier() + 100.0)
        assert np.isclose(np.linalg.norm(c), 1.0)

    def test_ncc_self_is_one(self):
        c = probe.carrier(_fixed_carrier())
        assert np.isclose(probe.ncc(c, c), 1.0)

    def test_ncc_mismatched_shape_is_incomparable(self):
        a = probe.carrier(np.random.default_rng(1).standard_normal((8, 8)))
        b = probe.carrier(np.random.default_rng(2).standard_normal((16, 16)))
        with np.testing.assert_raises_regex(ValueError, "geometry"):
            probe.ncc(a, b)


class TestConsistency:
    def test_shared_carrier_correlates_high(self):
        pattern = _fixed_carrier()
        carriers = [probe.carrier(_fill_with_carrier(b, pattern, s)) for s, b in enumerate((10, 60, 120, 200))]
        assert probe.mean_pairwise_ncc(carriers) > 0.8

    def test_random_fills_near_zero(self):
        rng = np.random.default_rng(0)
        carriers = [probe.carrier(rng.standard_normal(_SHAPE)) for _ in range(5)]
        assert abs(probe.mean_pairwise_ncc(carriers)) < 0.2

    def test_random_baseline_near_zero(self):
        assert abs(probe.random_baseline(_SHAPE, 6)) < 0.2


class TestRemoval:
    def test_removed_carrier_collapses_correlation(self):
        pattern = _fixed_carrier()
        pos = [probe.carrier(_fill_with_carrier(b, pattern, s)) for s, b in enumerate((20, 90, 160))]
        tmpl = probe.template(pos)
        # "Cleaned" fills keep the base + noise but lose the shared pattern.
        rng = np.random.default_rng(99)
        cleaned = [probe.carrier(b + 0.05 * rng.standard_normal(_SHAPE)) for b in (20, 90, 160)]
        pos_corr = float(np.mean([probe.ncc(c, tmpl) for c in pos]))
        cleaned_corr = float(np.mean([probe.ncc(c, tmpl) for c in cleaned]))
        assert pos_corr > 0.8
        assert cleaned_corr < 0.2


class TestCli:
    def _solid_fill_png(self, tmp_path: Path, name: str, base: int, pattern: np.ndarray, seed: int) -> Path:
        arr = np.clip(_fill_with_carrier(base, pattern, seed), 0, 255).astype(np.uint8)
        path = tmp_path / name
        Image.fromarray(np.stack([arr] * 3, axis=2)).save(path)
        return path

    def test_consistency_command_runs(self, tmp_path: Path):
        pattern = _fixed_carrier()
        paths = [str(self._solid_fill_png(tmp_path, f"f{i}.png", b, pattern, i)) for i, b in enumerate((40, 120, 200))]
        result = CliRunner().invoke(probe.cli, ["consistency", *paths])
        assert result.exit_code == 0, result.output
        assert "mean pairwise NCC" in result.output

    def test_consistency_needs_two_images(self, tmp_path: Path):
        path = str(self._solid_fill_png(tmp_path, "only.png", 100, _fixed_carrier(), 0))
        result = CliRunner().invoke(probe.cli, ["consistency", path])
        assert result.exit_code != 0

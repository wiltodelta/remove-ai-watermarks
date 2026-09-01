from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_periodic_tile_ablation as ablation
from synthid_periodic_tile import fold_residual_template, unit_tile
from synthid_periodic_tile_probe import PeriodicTileModel
from synthid_phase_carrier import PhaseCarrierModel


def test_exact_sign_test_detects_one_sided_direction() -> None:
    assert ablation.exact_sign_test(30, 0) == pytest.approx(1.862645149230957e-9)
    assert ablation.exact_sign_test(0, 0) == 1.0


def test_control_templates_are_norm_matched_and_random_control_is_orthogonal() -> None:
    rng = np.random.default_rng(7)
    template, _ = unit_tile(rng.normal(size=(8, 8, 3)))

    controls = ablation.control_templates(template, seed=11)

    assert set(controls) == {"aligned", "shifted", "orthogonal_random"}
    assert all(np.linalg.norm(control) == pytest.approx(1.0) for control in controls.values())
    assert np.sum(controls["orthogonal_random"] * template) == pytest.approx(0.0, abs=1e-12)
    assert not np.array_equal(controls["shifted"], template)


def test_aligned_subtraction_controls_both_synthetic_representations(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    raw_tile = rng.normal(size=(8, 8, 3))
    raw_tile -= np.mean(raw_tile, axis=(0, 1), keepdims=True)
    raw_tile, _ = unit_tile(raw_tile)
    source = np.clip(np.rint(128.0 + 24.0 * np.tile(raw_tile, (8, 8, 1))), 0, 255).astype(np.uint8)
    source_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_path)

    folded = fold_residual_template(
        source,
        tile_height=8,
        tile_width=8,
        denoise_sigma=1.0,
    )
    template, expected_norm = unit_tile(folded)
    tile_model = PeriodicTileModel(
        height=64,
        width=64,
        tile_height=8,
        tile_width=8,
        denoise_sigma=1.0,
        template=template,
        expected_norm=expected_norm,
    )

    spectra = np.stack([np.fft.rfft2(source[:, :, channel]) for channel in range(3)], axis=2)
    magnitude = np.abs(spectra)
    magnitude[0, 0, :] = 0.0
    row, column, channel = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    phase_model = PhaseCarrierModel(
        height=64,
        width=64,
        rows=np.asarray([row], dtype=np.int32),
        columns=np.asarray([column], dtype=np.int32),
        channels=np.asarray([channel], dtype=np.int8),
        phases=np.asarray([np.angle(spectra[row, column, channel])]),
        weights=np.asarray([1.0]),
        expected_magnitudes=np.asarray([magnitude[row, column, channel]]),
    )

    report = ablation.run_ablation(
        [source_path],
        tile_model=tile_model,
        phase_model=phase_model,
        tile_threshold=0.5,
        phase_threshold=0.5,
        active_threshold=0.0,
        strengths=(1.0, 2.0),
        phase_strength=2.0,
        seed=20260810,
    )

    assert report["original"] == {"tile_accepted": 1, "phase_accepted": 1}
    aligned = next(row for row in report["phase_summaries"] if row["control"] == "aligned")
    assert aligned["accepted"] == 0
    comparisons = {(row["aligned_minus"], row["metric"]): row for row in report["paired_comparisons"]}
    assert comparisons[("shifted", "tile_delta")]["difference"]["median"] < 0.0
    assert comparisons[("orthogonal_random", "tile_delta")]["difference"]["median"] < 0.0


def test_phase_strength_must_be_part_of_sweep() -> None:
    template = np.zeros((8, 8, 3), dtype=np.float64)
    template[0, 0, 0] = 1.0
    tile_model = PeriodicTileModel(64, 64, 8, 8, 1.0, template, 1.0)
    phase_model = PhaseCarrierModel(
        64,
        64,
        np.asarray([1]),
        np.asarray([1]),
        np.asarray([0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([1.0]),
    )

    with pytest.raises(ValueError, match="phase strength"):
        ablation.run_ablation(
            [Path("unused.png")],
            tile_model=tile_model,
            phase_model=phase_model,
            tile_threshold=0.0,
            phase_threshold=0.0,
            active_threshold=0.0,
            strengths=(1.0,),
            phase_strength=2.0,
            seed=1,
        )

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_affine_lattice_probe as probe
from synthid_runtime._synthid_confirmation import RegisteredConfirmationComponents


def test_webp_lossless_round_trip_preserves_pixels() -> None:
    rng = np.random.default_rng(20260817)
    pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    restored = probe._webp_round_trip(pixels, 101)

    assert np.array_equal(restored, pixels)


@pytest.fixture(scope="module")
def periodic_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260814)
    template = rng.normal(0.0, 1.0, (16, 16, 3))
    template -= np.mean(template, axis=(0, 1), keepdims=True)
    template /= np.linalg.norm(template)
    coarse = rng.normal(0.0, 8.0, (16, 16, 3)).astype(np.float32)
    background = cv2.resize(coarse, (1024, 1024), interpolation=cv2.INTER_CUBIC) + 128.0
    carrier = np.tile(template, (64, 64, 1)) * 3.0
    pixels = np.clip(np.rint(background + carrier), 0, 255).astype(np.uint8)
    return pixels, template


def _score(pixels: np.ndarray, template: np.ndarray) -> probe.LatticeScore:
    return probe.score_lattice(
        pixels,
        template,
        periods=np.arange(12.0, 20.01, 0.25),
        rotations_degrees=np.asarray([-1.0, 0.0, 1.0]),
        patch_size=256,
        grid_size=4,
        harmonic_count=12,
    )


def test_period_alias_candidates_include_base_and_half_period_neighbors() -> None:
    periods = np.arange(7.5, 24.501, 0.1)
    rotations = np.zeros_like(periods)
    base_index = int(np.argmin(np.abs(periods - 19.2)))

    candidates = probe._period_alias_candidate_indices(periods, rotations, [base_index])

    assert [periods[index] for index in candidates] == pytest.approx([19.1, 19.2, 19.3, 9.5, 9.6, 9.7])


def test_split_lattice_recovers_periodic_carrier(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture

    result = _score(pixels, template)

    assert result.selected_period == pytest.approx(16.0, abs=0.25)
    assert result.selected_rotation_degrees == 0.0
    assert result.confirmation_coherence > 0.9
    assert result.joint_coherence > 0.9
    assert result.joint_codeword > 0.8
    assert result.unknown_codeword_confirmation > 0.8
    assert result.unknown_codeword_fixed_confirmation > 0.8
    assert result.unknown_codeword_fixed_all > 0.8
    assert result.unknown_codeword_excess_p99 > 0.0
    assert result.joint_amplitude > 0.8
    assert result.joint_whitened_match > 0.8
    assert result.canonical_template_score > 0.8
    assert result.canonical_registered_template_score > 0.8
    assert result.confirmation_excess_p99 > 0.0
    assert result.selection_patches == result.confirmation_patches == 8


def test_split_lattice_rejects_independent_noise(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    _pixels, template = periodic_fixture
    rng = np.random.default_rng(20260815)
    noise = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)

    result = _score(noise, template)

    assert result.confirmation_coherence < 0.8
    assert result.joint_coherence < 0.8
    assert result.joint_codeword < 0.8
    assert result.unknown_codeword_confirmation < 0.2
    assert result.unknown_codeword_fixed_confirmation < 0.2
    assert result.unknown_codeword_fixed_all < 0.2
    assert result.joint_amplitude < 0.2
    assert result.joint_whitened_match < 0.2
    assert result.canonical_template_score < 0.2
    assert result.canonical_registered_template_score < 0.2
    assert result.confirmation_excess_p99 < 0.0


def test_split_lattice_tracks_resized_period(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    resized = cv2.resize(pixels, (819, 819), interpolation=cv2.INTER_CUBIC)

    result = probe.score_lattice(
        resized,
        template,
        periods=np.arange(7.5, 24.501, 0.1),
        rotations_degrees=np.asarray([0.0]),
        patch_size=192,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_period == pytest.approx(12.8, abs=0.3)
    assert result.confirmation_coherence > 0.8
    assert result.joint_amplitude > 0.8
    assert result.joint_whitened_match > 0.8
    assert result.unknown_codeword_confirmation > 0.8
    assert result.unknown_codeword_fixed_confirmation > 0.8
    assert result.unknown_codeword_fixed_all > 0.8


def test_split_lattice_tracks_octave_aliased_resize(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    resized = cv2.resize(pixels, (614, 614), interpolation=cv2.INTER_AREA)

    result = probe.score_lattice(
        resized,
        template,
        periods=np.arange(7.5, 24.501, 0.1),
        rotations_degrees=np.asarray([0.0]),
        patch_size=192,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_period == pytest.approx(9.6, abs=0.15)
    assert result.canonical_template_score > 0.4


def test_same_image_period_null_prefers_the_carrier_period(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    pixels, template = periodic_fixture

    correct = probe.score_same_image_period_null(pixels, template, 16.0, harmonic_count=12)
    off_period = probe.score_same_image_period_null(pixels, template, 15.0, harmonic_count=12)

    assert correct.joint_excess > 0.2
    assert correct.joint_excess > off_period.joint_excess
    assert correct.off_period_count == len(probe.SAME_IMAGE_NULL_OFFSETS)


def test_patch_shift_consensus_confirms_global_carrier_phase(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    pixels, template = periodic_fixture
    rng = np.random.default_rng(20260818)
    noise = rng.integers(0, 256, pixels.shape, dtype=np.uint8)

    carrier = probe.score_patch_shift_consensus(pixels, template, 16.0)
    control = probe.score_patch_shift_consensus(noise, template, 16.0)

    assert carrier.joint_trimmed_z > control.joint_trimmed_z
    assert carrier.joint_support_fraction == 1.0
    assert carrier.selection_patches == carrier.confirmation_patches == 8


def test_patch_shift_recovery_uses_frozen_mechanism_gates() -> None:
    baseline = {
        "amplitude_margin": 0.8,
        "high_band_margin": 1.0,
        "periods_agree": True,
        "confirmation_passes": True,
        "joint_trimmed_z": 2.5,
    }

    assert probe.patch_shift_recovery_passes(**baseline)
    for field, failed_value in (
        ("amplitude_margin", 0.449),
        ("high_band_margin", 0.449),
        ("periods_agree", False),
        ("confirmation_passes", False),
        ("joint_trimmed_z", 2.499),
    ):
        candidate = {**baseline, field: failed_value}
        assert not probe.patch_shift_recovery_passes(**candidate)
    assert not probe.patch_shift_recovery_passes(**{**baseline, "amplitude_margin": 0.99, "high_band_margin": 0.99})


def test_opponent_registration_recovers_resampled_carrier(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    pixels, template = periodic_fixture
    resized = cv2.resize(pixels, (717, 717), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(20260819)
    noise = rng.integers(0, 256, resized.shape, dtype=np.uint8)
    periods = np.arange(10.0, 12.41, 0.05)

    carrier = probe.score_opponent_registered(resized, template, periods=periods)
    control = probe.score_opponent_registered(noise, template, periods=periods)

    assert carrier.selected_period == pytest.approx(11.2, abs=0.1)
    assert carrier.decision_score > 1.0
    assert control.decision_score < 1.0


def test_split_lattice_aligns_cyclic_carrier_phase(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    shifted = np.roll(pixels, shift=(3, 5), axis=(0, 1))

    result = _score(shifted, template)

    assert (result.selected_shift_y, result.selected_shift_x) == (3, 5)
    assert (result.amplitude_shift_y, result.amplitude_shift_x) == (13, 11)
    assert result.canonical_template_score < 0.2
    assert result.canonical_registered_template_score > 0.8
    assert result.joint_whitened_match < 0.4
    assert result.unknown_codeword_confirmation > 0.8
    assert result.unknown_codeword_fixed_confirmation > 0.8
    assert result.unknown_codeword_fixed_all > 0.8


def test_split_lattice_recovers_cropped_carrier_phase(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    cropped = pixels[37:, 53:]

    result = probe.score_lattice(
        cropped,
        template,
        periods=np.asarray([16.0]),
        rotations_degrees=np.asarray([0.0]),
        patch_size=256,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_period == pytest.approx(16.0, abs=0.25)
    assert result.joint_coherence > 0.8
    assert result.unknown_codeword_fixed_all > 0.8
    assert result.canonical_template_score < 0.2
    assert result.canonical_registered_template_score > 0.8


def test_orientation_bank_recovers_right_angle_rotation(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    rotated_clockwise = np.rot90(pixels, k=-1)

    result = probe.score_orientation_bank(
        rotated_clockwise,
        template,
        periods=np.asarray([16.0]),
        rotations_degrees=np.asarray([0.0]),
        patch_size=256,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_orientation_degrees == 90
    assert result.joint_amplitude > 0.8
    assert result.canonical_template_score > 0.8


def test_dihedral_bank_recovers_horizontal_reflection(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture

    result = probe.score_dihedral_bank(
        np.fliplr(pixels),
        template,
        periods=np.asarray([16.0]),
        rotations_degrees=np.asarray([0.0]),
        patch_size=256,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_orientation_degrees == 0
    assert result.selected_horizontal_reflection is True
    assert result.joint_amplitude > 0.8
    assert result.canonical_template_score > 0.8


def test_deskew_bank_recovers_small_rotation(periodic_fixture: tuple[np.ndarray, np.ndarray]) -> None:
    pixels, template = periodic_fixture
    rotated = probe._rotate_fixed_canvas(pixels, 1.5)

    result = probe.score_deskew_bank(
        rotated,
        template,
        periods=np.asarray([16.0]),
        deskew_degrees=np.asarray([-2.0, -1.5, -1.0]),
        patch_size=256,
        grid_size=4,
        harmonic_count=12,
    )

    assert result.selected_deskew_degrees == -1.5
    assert result.joint_amplitude > 0.6
    assert result.canonical_template_score > 0.6
    assert result.deskew_direct_joint_match > 0.4


def test_registered_period_mode_uses_runtime_selected_period(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels, template = periodic_fixture
    runner = CliRunner()
    with runner.isolated_filesystem():
        np.savez("template.npz", template=template)
        Image.fromarray(pixels).save("image.png")
        components = probe.RegisteredComponents(
            raw_score=0.4,
            amplitude_threshold=0.2,
            selected_period=16.0,
            spectral_period=16.0,
            high_band_score=0.15,
            confirmation=RegisteredConfirmationComponents(
                period=16.0,
                joint_coherence=0.5,
                joint_amplitude=0.2,
                unknown_codeword_fixed_confirmation=0.5,
                selection_patches=8,
                confirmation_patches=8,
            ),
        )
        monkeypatch.setattr(probe, "registered_components", lambda *_args: components)

        result = runner.invoke(
            probe.main,
            [
                "template.npz",
                "image.png",
                "--registered-period",
                "--same-image-null",
                "--patch-shift-consensus",
                "--opponent-registered",
                "--report-out",
                "report.json",
            ],
        )

        assert result.exit_code == 0, result.output
        report = json.loads(Path("report.json").read_text(encoding="utf-8"))
        assert report["registered_period"] is True
        assert report["same_image_null"] is True
        assert report["patch_shift_consensus"] is True
        assert report["opponent_registered"] is True
        assert report["records"][0]["registered"]["selected_period"] == 16.0
        assert report["records"][0]["registered"]["decision_score"] == 2.0
        assert report["records"][0]["score"]["selected_period"] == 16.0
        assert report["records"][0]["same_image_null"]["joint_excess"] > 0.2
        assert report["records"][0]["patch_shift_consensus"]["joint_support_fraction"] == 1.0
        assert report["records"][0]["opponent_registered"]["decision_score"] > 1.0


def test_registered_confirmation_uses_frozen_period_aware_gates(
    periodic_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    pixels, template = periodic_fixture
    baseline = _score(pixels, template)
    generic = replace(
        baseline,
        selected_period=16.0,
        joint_coherence=0.30,
        joint_amplitude=0.0,
    )

    assert probe.registered_confirmation_passes(generic)
    assert not probe.registered_confirmation_passes(replace(generic, selected_period=9.99))
    assert not probe.registered_confirmation_passes(replace(generic, joint_coherence=0.299))
    assert not probe.registered_confirmation_passes(replace(generic, joint_amplitude=-0.001))
    assert not probe.registered_confirmation_passes(
        replace(generic, selected_period=18.28, unknown_codeword_fixed_confirmation=0.129)
    )
    assert probe.registered_confirmation_passes(
        replace(generic, selected_period=18.28, unknown_codeword_fixed_confirmation=0.13)
    )
    assert not probe.registered_confirmation_passes(replace(generic, selected_period=19.14, joint_coherence=0.399))
    assert probe.registered_confirmation_passes(replace(generic, selected_period=19.14, joint_coherence=0.40))
    assert not probe.registered_confirmation_passes(
        replace(generic, selected_period=21.31, unknown_codeword_fixed_confirmation=0.019)
    )
    assert probe.registered_confirmation_passes(
        replace(generic, selected_period=21.31, unknown_codeword_fixed_confirmation=0.02)
    )


def test_fixed_candidate_uses_frozen_precision_threshold() -> None:
    assert not probe.fixed_candidate_passes(0.279999)
    assert probe.fixed_candidate_passes(0.28)
    assert not probe.fixed_candidate_passes(float("nan"))

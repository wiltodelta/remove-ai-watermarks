"""Check alignment direction, donor-only colors and protected geometry."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from remove_ai_watermarks._internal import image_alignment as study


def scene():
    rng = np.random.default_rng(12)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (256, 256), dtype=np.uint8), (0, 0), 3)
    return np.repeat(gray[..., None], 3, axis=2)


def test_alignment_recovers_a_known_translation_without_copying_source_colors():
    reference = scene()
    donor = cv2.warpAffine(
        reference, np.float32([[1, 0, 5], [0, 1, -3]]), (256, 256), borderMode=cv2.BORDER_REFLECT_101
    )
    donor = np.clip(donor.astype(int) + 10, 0, 255).astype(np.uint8)
    output, flow = study.align_regenerated(reference, donor)
    assert (
        np.mean((output[32:-32, 32:-32].astype(float) - reference[32:-32, 32:-32]) ** 2)
        < np.mean((donor[32:-32, 32:-32].astype(float) - reference[32:-32, 32:-32]) ** 2) * 0.65
    )
    assert np.median(flow[32:-32, 32:-32, 0]) > 4
    assert np.median(flow[32:-32, 32:-32, 1]) < -2
    assert np.mean(output.astype(float) - reference) > 8


def test_constant_donor_cannot_acquire_reference_texture_or_colors():
    donor = np.full((256, 256, 3), (60, 90, 120), np.uint8)
    output, _ = study.align_regenerated(scene(), donor)
    assert np.array_equal(output, donor)


def test_faces_remain_exact_and_inputs_unchanged():
    reference = scene()
    donor = np.roll(reference, 7, axis=1)
    originals = reference.copy(), donor.copy()
    output, flow = study.align_regenerated(reference, donor, ((70, 60, 170, 190),))
    assert np.array_equal(output[60:190, 70:170], donor[60:190, 70:170])
    assert not np.any(flow[60:190, 70:170])
    assert np.array_equal(reference, originals[0])
    assert np.array_equal(donor, originals[1])


@pytest.mark.parametrize("discontinuous", [False, True])
def test_displacement_and_gradient_are_bounded(monkeypatch, discontinuous):
    class Estimator:
        def calc(self, reference, donor, initial):
            result = np.full((*reference.shape, 2), 100, np.float32)
            if discontinuous:
                result[:, : reference.shape[1] // 2] = -100
            return result

    monkeypatch.setattr(study.cv2, "DISOpticalFlow", SimpleNamespace(create=lambda preset: Estimator()))
    reference = scene()
    protected = ((60, 60, 120, 120),) if discontinuous else ()
    _, flow = study.align_regenerated(reference, np.roll(reference, 1, axis=1), protected)
    assert np.max(np.linalg.norm(flow, axis=2)) <= 16.0001
    derivatives = [np.gradient(flow[..., axis]) for axis in range(2)]
    norm = np.sqrt(sum(component**2 for pair in derivatives for component in pair))
    assert np.max(norm) <= 0.35001


def test_identical_inputs_remain_exact_and_independent():
    reference = scene()
    output, flow = study.align_regenerated(reference, reference)
    assert np.array_equal(output, reference)
    assert not np.shares_memory(output, reference)
    assert not np.any(flow)


@pytest.mark.parametrize(("mode", "interpolation"), [("linear", cv2.INTER_LINEAR), ("lanczos", cv2.INTER_LANCZOS4)])
def test_resampling_uses_only_donor_samples_with_the_requested_kernel(mode, interpolation):
    reference = scene()
    donor = np.roll(reference, 5, axis=1)
    output, flow = study.align_regenerated(reference, donor, resampling=mode)
    yy, xx = np.indices(reference.shape[:2], dtype=np.float32)
    expected = cv2.remap(donor, xx + flow[..., 0], yy + flow[..., 1], interpolation, borderMode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(output, expected)


def test_unknown_resampling_fails_before_identity_shortcut():
    reference = scene()
    with pytest.raises(ValueError, match="Unknown alignment resampling"):
        study.align_regenerated(reference, reference, resampling="unknown")


@pytest.mark.parametrize("box", [(0, 0, 300, 10), (5, 5, 4, 6), (1.5, 0, 8, 8)])
def test_invalid_regions_fail_before_alignment(box):
    with pytest.raises(ValueError, match=r"[Pp]rotected box"):
        study.align_regenerated(scene(), scene(), (box,))


def test_weighted_geometry_uses_one_warp_instead_of_blending_rgb():
    donor = scene()
    flow = np.full((256, 256, 2), (5, -3), dtype=np.float32)
    weights = np.full((256, 256), 0.5, np.float32)
    output, weighted = study.attenuate_regenerated(donor, flow, weights)
    yy, xx = np.indices((256, 256), dtype=np.float32)
    expected = cv2.remap(donor, xx + 2.5, yy - 1.5, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(output, expected)
    assert np.all(weighted == (2.5, -1.5))
    assert np.all(flow == (5, -3))
    assert np.all(weights == 0.5)


def test_zero_weights_preserve_exact_donor():
    donor = scene()
    output, flow = study.attenuate_regenerated(
        donor, np.ones((256, 256, 2), np.float32), np.zeros((256, 256), np.float32)
    )
    assert np.array_equal(output, donor)
    assert not np.any(flow)


def test_weighted_flow_stays_nonfolding_and_preserves_protected_pixels():
    donor = scene()
    flow = np.full((256, 256, 2), (8, -4), np.float32)
    flow[80:160, 80:160] = 0
    weights = np.ones((256, 256), np.float32)
    weights[:, :128] = 0
    output, weighted = study.attenuate_regenerated(donor, flow, weights)
    assert np.array_equal(output[80:160, 80:160], donor[80:160, 80:160])
    derivatives = [np.gradient(weighted[..., axis]) for axis in range(2)]
    norm = np.sqrt(sum(component**2 for pair in derivatives for component in pair))
    assert np.max(norm) <= 0.35001


@pytest.mark.parametrize("kind", ["negative", "overshoot", "nan", "shape", "integer", "large_flow"])
def test_bad_weighted_geometry_is_rejected(kind):
    donor = scene()
    flow = np.zeros((256, 256, 2), np.float32)
    weights = np.ones((256, 256), np.float32)
    if kind == "negative":
        weights[0, 0] = -0.01
    elif kind == "overshoot":
        weights[0, 0] = 1.01
    elif kind == "nan":
        flow[0, 0] = np.nan
    elif kind == "shape":
        weights = weights[:-1]
    elif kind == "integer":
        weights = weights.astype(np.uint8)
    elif kind == "large_flow":
        flow[0, 0] = (17, 0)
    with pytest.raises(ValueError, match=r"[Ww]eight|[Dd]isplacement|dimensions"):
        study.attenuate_regenerated(donor, flow, weights)

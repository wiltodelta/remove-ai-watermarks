"""Check coarse donor-error weights independently of the image warp."""

import numpy as np

from remove_ai_watermarks._internal import image_alignment as study


def test_uses_one_scalar_weight_for_all_color_channels():
    baseline = np.zeros((64, 64, 3), np.uint8)
    candidate = np.full_like(baseline, 200)
    source = np.tile(np.array([120, 80, 100], np.uint8), (64, 64, 1))
    weights = study.local_error_weights(source, baseline, candidate)
    assert np.allclose(weights, 0.5)


def test_source_checkerboard_does_not_become_pixelwise_weights():
    source = np.repeat(((np.indices((64, 64)).sum(axis=0) % 2) * 200).astype(np.uint8)[..., None], 3, axis=2)
    baseline = np.zeros_like(source)
    candidate = np.full_like(source, 200)
    weights = study.local_error_weights(source, baseline, candidate, sigma=16)
    assert np.max(weights) - np.min(weights) < 0.001
    assert np.allclose(weights, 0.5, atol=0.001)


def test_local_donor_choice_has_a_smooth_transition():
    baseline = np.zeros((256, 512, 3), np.uint8)
    candidate = np.full_like(baseline, 200)
    source = baseline.copy()
    source[:, 256:] = 200
    weights = study.local_error_weights(source, baseline, candidate, sigma=8)
    assert np.all(weights[:, :100] == 0)
    assert np.allclose(weights[:, 412:], 1)
    assert 0 < weights[128, 255] < 1
    assert np.max(np.abs(np.diff(weights, axis=1))) < 0.05


def test_identical_donors_have_zero_displacement_weight():
    donor = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert not np.any(study.local_error_weights(np.zeros_like(donor), donor, donor))


def test_final_bounds_survive_backend_rounding(monkeypatch):
    gaussian = study.cv2.GaussianBlur

    def rounded(values, *args, **kwargs):
        output = gaussian(values, *args, **kwargs)
        if np.all(values == 1):
            output.fill(np.nextafter(np.float32(1), np.float32(2)))
        return output

    monkeypatch.setattr(study.cv2, "GaussianBlur", rounded)
    baseline = np.zeros((64, 64, 3), np.uint8)
    candidate = np.full_like(baseline, 100)
    weights = study.local_error_weights(np.full_like(baseline, 255), baseline, candidate)
    assert np.min(weights) >= 0
    assert np.max(weights) <= 1

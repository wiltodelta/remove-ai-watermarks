import numpy as np

from remove_ai_watermarks.humanizer import apply_analog_humanizer


def test_humanizer_does_not_modify_original_if_disabled():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50, 50] = [100, 150, 200]
    org_img = img.copy()

    # grain=0, shift=0 means disabled essentially. But wait, apply_analog_humanizer currently applies chromatic shift even if grain=0.
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=0)
    assert np.array_equal(result, org_img)


def test_chromatic_shift():
    # Only green channel is centered, red/blue should shift.
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[2, 2] = [255, 255, 255]  # B, G, R

    # shift=1
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=1)

    # G (index 1) stays at [2,2]
    assert result[2, 2, 1] == 255
    # B (index 0) shifted right (+1 axis 1) -> [2, 3]
    assert result[2, 3, 0] == 255
    # R (index 2) shifted left (-1 axis 1) -> [2, 1]
    assert result[2, 1, 2] == 255


def test_grain_intensity():
    # Gray image
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Add strong noise
    result = apply_analog_humanizer(img, grain_intensity=10.0, chromatic_shift=0)

    # Image should no longer be purely 128
    unique_vals = np.unique(result)
    assert len(unique_vals) > 5

    # Mean should roughly be 128
    assert 126 < np.mean(result) < 130


def test_invalid_shape():
    # Missing color channel
    img = np.zeros((100, 100), dtype=np.uint8)
    img[0, 0] = 50
    result = apply_analog_humanizer(img)
    assert np.array_equal(img, result)

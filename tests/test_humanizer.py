import numpy as np
import pytest

from remove_ai_watermarks.humanizer import apply_analog_humanizer, unsharp_mask


def test_humanizer_does_not_modify_original_if_disabled():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50, 50] = [100, 150, 200]
    org_img = img.copy()

    # grain=0, shift=0 means disabled — result should match original.
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


def test_chromatic_shift_does_not_wrap_opposite_edge():
    # On a horizontal gradient (dark left, bright right), a circular np.roll
    # would wrap the bright right edge into the R channel's left border and the
    # dark left edge into the B channel's right border, producing a colored
    # fringe. After the fix the border columns must replicate their own edge.
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)
    gray = np.broadcast_to(ramp, (32, 64))
    img = np.stack([gray, gray, gray], axis=2).copy()  # B, G, R

    shift = 3
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=shift)

    # B (index 0) rolled right -> its left border must stay dark (near 0),
    # NOT wrap the bright right edge.
    assert result[:, :shift, 0].max() < 60
    # R (index 2) rolled left -> its right border must stay bright (near 255),
    # NOT wrap the dark left edge.
    assert result[:, -shift:, 2].min() > 195


@pytest.mark.parametrize(("width", "shift"), [(1, 1), (3, 3), (3, 5), (5, 10)])
def test_chromatic_shift_wider_than_image_no_crash(width: int, shift: int):
    """Regression: a chromatic_shift >= image width left the edge-replication slices
    empty and crashed the broadcast (ValueError). The shift must clamp to width-1."""
    img = np.full((4, width, 3), 120, np.uint8)
    result = apply_analog_humanizer(img, grain_intensity=0.0, chromatic_shift=shift)
    assert result.shape == img.shape


def test_unsharp_disabled_returns_unchanged_copy():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    img[10, 10] = [100, 150, 200]
    result = unsharp_mask(img, amount=0.0)
    assert np.array_equal(result, img)
    assert result is not img  # a fresh copy, never the same array


def test_unsharp_overshoots_at_an_edge():
    # A vertical step (left 100, right 150). Unsharp masking overshoots at the
    # boundary, pushing pixels above the bright level and below the dark level.
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    img[:, 10:] = 150
    result = unsharp_mask(img, amount=1.0, sigma=1.5)
    assert int(result.max()) > 150  # bright-side overshoot
    assert int(result.min()) < 100  # dark-side undershoot


def test_unsharp_preserves_shape_and_dtype():
    img = np.full((15, 25, 3), 120, dtype=np.uint8)
    result = unsharp_mask(img, amount=0.6)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_unsharp_flat_image_is_a_noop():
    # No edges -> blur equals the image -> unsharp cancels to the original.
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    result = unsharp_mask(img, amount=0.8, sigma=1.0)
    assert np.array_equal(result, img)


class TestAdaptivePolish:
    """Adaptive polish: target the reference's detail level, sparing text/edges."""

    def test_noop_when_already_sharp(self):
        from remove_ai_watermarks.humanizer import adaptive_polish

        rng = np.random.default_rng(1)
        sharp = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)  # high detail
        soft_ref = np.full((120, 120, 3), 128, dtype=np.uint8)  # flat -> low target
        out = adaptive_polish(sharp, soft_ref)
        assert np.array_equal(out, sharp)  # current >= target -> unchanged copy

    def test_sharpens_a_soft_image_toward_reference(self):
        import cv2

        from remove_ai_watermarks.humanizer import _laplacian_variance, adaptive_polish

        rng = np.random.default_rng(2)
        reference = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)  # very high detail
        soft = cv2.GaussianBlur(reference, (0, 0), sigmaX=4.0)  # blurred -> low detail
        out = adaptive_polish(soft, reference, seed=0)
        assert _laplacian_variance(out) > _laplacian_variance(soft)  # moved toward the target

    def test_mask_spares_edges(self):
        from remove_ai_watermarks.humanizer import _smooth_grain_mask

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[:, 50:] = 30  # a hard vertical edge down the middle
        mask = _smooth_grain_mask(img)
        # Flat far-left region keeps grain; the column at the edge is suppressed.
        assert mask[:, :15].mean() > mask[:, 45:55].mean()

    def test_deterministic_with_seed(self):
        import cv2

        from remove_ai_watermarks.humanizer import adaptive_polish

        rng = np.random.default_rng(3)
        reference = rng.integers(0, 256, (140, 140, 3), dtype=np.uint8)
        soft = cv2.GaussianBlur(reference, (0, 0), sigmaX=3.0)
        a = adaptive_polish(soft, reference, seed=7)
        b = adaptive_polish(soft, reference, seed=7)
        assert np.array_equal(a, b)

    def test_the_noop_reports_itself_instead_of_returning_silently(self):
        from remove_ai_watermarks.humanizer import adaptive_polish

        rng = np.random.default_rng(11)
        sharp = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)
        soft_ref = np.full((120, 120, 3), 128, dtype=np.uint8)
        said: list[str] = []
        out = adaptive_polish(sharp, soft_ref, on_skip=said.append)
        assert np.array_equal(out, sharp)
        assert len(said) == 1
        assert "Laplacian variance" in said[0]

    def test_a_polish_that_does_something_says_nothing(self):
        import cv2

        from remove_ai_watermarks.humanizer import adaptive_polish

        rng = np.random.default_rng(12)
        reference = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
        soft = cv2.GaussianBlur(reference, (0, 0), sigmaX=4.0)
        said: list[str] = []
        adaptive_polish(soft, reference, seed=0, on_skip=said.append)
        assert said == []

    def test_grain_before_the_polish_is_what_silences_it(self):
        # The composition bug this callback exists for: humanize first raises the
        # measured detail level past the reference's, so the polish self-limits to
        # nothing and the grain is all that survives.
        import cv2

        from remove_ai_watermarks.humanizer import adaptive_polish, apply_analog_humanizer

        rng = np.random.default_rng(13)
        # A reference with a REAL photo's detail level, not pure noise: a target of
        # thousands would swallow any grain and hide the interaction.
        raw = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
        reference = cv2.GaussianBlur(raw, (0, 0), sigmaX=2.0)
        soft = cv2.GaussianBlur(raw, (0, 0), sigmaX=4.0)
        grainy = apply_analog_humanizer(soft, grain_intensity=12.0, chromatic_shift=1)
        said: list[str] = []
        out = adaptive_polish(grainy, reference, seed=0, on_skip=said.append)
        assert np.array_equal(out, grainy)  # the polish contributed nothing
        assert len(said) == 1

    def test_all_edges_reference_grain_mask_near_zero(self):
        # An all-high-frequency target: _smooth_grain_mask suppresses edges, so the grain
        # mask is ~all-zero (grain adds nothing) -- adaptive_polish must still return a
        # valid same-shape image, not crash on the empty-mask branch.
        import cv2

        from remove_ai_watermarks.humanizer import _smooth_grain_mask, adaptive_polish

        rng = np.random.default_rng(5)
        edges = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)  # all high-frequency
        assert _smooth_grain_mask(edges).mean() < _smooth_grain_mask(np.full((120, 120, 3), 128, np.uint8)).mean()
        soft = cv2.GaussianBlur(edges, (0, 0), sigmaX=3.0)
        out = adaptive_polish(soft, edges, seed=0)
        assert out.shape == soft.shape
        assert out.dtype == np.uint8

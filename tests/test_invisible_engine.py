"""Tests for the invisible watermark engine (unit tests, no GPU required)."""

from __future__ import annotations

from remove_ai_watermarks.invisible_engine import InvisibleEngine, _target_size, is_available


class TestIsAvailable:
    """Tests for dependency checking."""

    def test_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)

    def test_available_reflects_dependencies(self):
        """is_available() is True iff torch + diffusers (the gpu extra) import.

        Must not assume the full stack: the core+dev CI env has no diffusers.
        """
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_available() is expected


class TestInvisibleEngineInit:
    """Tests for InvisibleEngine construction (no GPU required)."""

    def test_default_model_id(self):
        # SDXL base became the default in May 2026 (defeats SynthID v2).
        assert InvisibleEngine.DEFAULT_MODEL_ID == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_ctrlregen_model_id(self):
        assert InvisibleEngine.CTRLREGEN_MODEL_ID == "yepengliu/ctrlregen"


class TestTargetSize:
    """Regression guard for the native-resolution decision (issues #10 / #15).

    max_resolution=0 must NOT downscale -- the forced downscale->upscale
    round-trip was the quality loss in #10, and downscaling at all let SynthID
    survive in #15 (the native SDXL pass at strength ~0.05 is what defeats it).
    """

    def test_native_default_no_downscale(self):
        # The default (0) means native resolution: no resize, regardless of size.
        assert _target_size(4096, 4096, 0) is None
        assert _target_size(123, 456, 0) is None

    def test_negative_cap_treated_as_native(self):
        assert _target_size(4096, 4096, -1) is None

    def test_cap_below_long_side_downscales(self):
        # 2000x1000, cap 1024 -> long side scaled to 1024, aspect preserved.
        assert _target_size(2000, 1000, 1024) == (1024, 512)

    def test_cap_uses_long_side_for_portrait(self):
        # Portrait: height is the long side, so it drives the ratio.
        assert _target_size(1000, 2000, 1024) == (512, 1024)

    def test_cap_at_or_above_long_side_no_downscale(self):
        # Already within the cap (and exactly equal) -> no resize.
        assert _target_size(800, 600, 1024) is None
        assert _target_size(1024, 768, 1024) is None

    def test_integer_truncation_matches_pil_call_site(self):
        # 1254x1254 (the gpt-image sample) capped at 1000: int(1254*1000/1254)=1000.
        assert _target_size(1254, 1254, 1000) == (1000, 1000)
        # Non-divisible ratio truncates toward zero like int() at the call site.
        assert _target_size(1000, 333, 500) == (500, 166)

    def test_extreme_aspect_ratio_clamps_short_side_to_one(self):
        # 5000x3 capped at 1024: int(3 * 1024/5000) = 0 would crash resize();
        # the short side must clamp to 1, never 0.
        assert _target_size(5000, 3, 1024) == (1024, 1)
        assert _target_size(3, 5000, 1024) == (1, 1024)

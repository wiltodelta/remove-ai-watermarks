"""Inpaint-fallback visible removal: method resolution, footprint masks, dispatch.

The inpaint path erases the mark footprint (MI-GAN when the ``migan`` extra is
installed, else cv2) instead of reverse-alpha, so a mark needs no captured alpha
map for removal (only for the footprint silhouette). ``auto`` is deterministic:
reverse-alpha for capture marks, inpaint for capture-less. These tests avoid any
ONNX model download by pinning the backend to cv2 via ``preferred_inpaint_backend``;
only pure cv2/numpy paths run.
"""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.doubao_engine import DoubaoEngine
from remove_ai_watermarks.gemini_engine import GeminiEngine


def _compose_textmark(engine, bg: float = 120.0, w: int = 1024, h: int = 1024):
    """Composite the engine's captured mark onto a flat ``bg`` at full opacity so
    the mark is detectable. Returns ``(watermarked_uint8, (ax, ay, gw, gh))``."""
    img = np.full((h, w, 3), float(bg), np.float32)
    block, (ax, ay, gw, gh) = engine._fixed_alpha_map(img)
    a = np.clip(block, 0.0, 0.99)[:, :, None]
    logo = np.array(engine.config.alpha_logo_bgr, np.float32)
    img[ay : ay + gh, ax : ax + gw] = img[ay : ay + gh, ax : ax + gw] * (1 - a) + logo * a
    return np.clip(img, 0, 255).astype(np.uint8), (ax, ay, gw, gh)


class TestResolveMethod:
    @pytest.mark.parametrize("explicit", ["reverse-alpha", "inpaint"])
    def test_explicit_passthrough(self, explicit: str) -> None:
        # capture mark: the explicit method passes through unchanged
        assert registry.resolve_removal_method(explicit, True) == explicit  # type: ignore[arg-type]
        # capture-less: reverse-alpha is impossible, so it collapses to inpaint
        expected = "inpaint" if explicit == "reverse-alpha" else explicit
        assert registry.resolve_removal_method(explicit, False) == expected  # type: ignore[arg-type]

    def test_auto_uses_reverse_alpha_for_capture_marks(self) -> None:
        # auto is deterministic and model-independent: reverse-alpha where a capture
        # exists (cleaner + lighter than inpaint), inpaint only where it does not.
        assert registry.resolve_removal_method("auto", True) == "reverse-alpha"

    def test_auto_uses_inpaint_for_capture_less(self) -> None:
        assert registry.resolve_removal_method("auto", False) == "inpaint"


class TestFootprintMask:
    def test_textmark_footprint_geometry(self) -> None:
        mask = DoubaoEngine().footprint_mask(np.full((1024, 1024, 3), 120, np.uint8))
        assert mask is not None
        assert mask.shape == (1024, 1024)
        assert mask.dtype == np.uint8
        assert mask.any()
        # Doubao sits bottom-right: the mask mass is in the bottom-right quadrant.
        ys, xs = np.where(mask > 0)
        assert ys.mean() > 512
        assert xs.mean() > 512

    def test_textmark_small_image_returns_none(self) -> None:
        assert DoubaoEngine().footprint_mask(np.full((20, 20, 3), 120, np.uint8)) is None

    def test_gemini_footprint_needs_detection_or_force(self) -> None:
        eng = GeminiEngine()
        clean = np.full((1024, 1024, 3), 128, np.uint8)
        assert eng.footprint_mask(clean) is None  # nothing detected -> no mask
        forced = eng.footprint_mask(clean, force=True)  # default sparkle slot
        assert forced is not None
        assert forced.any()


class TestInpaintDispatch:
    """Force the cv2 backend (patch preferred_inpaint_backend) so no ONNX model
    downloads; the dispatch/gating logic is backend-agnostic."""

    def test_clean_image_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "preferred_inpaint_backend", lambda: "cv2")
        img = np.full((1024, 1024, 3), 120, np.uint8)
        out, region = registry.get_mark("doubao").remove(img, method="inpaint")
        assert region is None
        assert np.array_equal(out, img)  # not detected, not forced -> no-op

    def test_forced_inpaint_edits_only_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "preferred_inpaint_backend", lambda: "cv2")
        img, (ax, ay, gw, gh) = _compose_textmark(DoubaoEngine())
        out, _ = registry.get_mark("doubao").remove(img, method="inpaint", force=True)
        assert not np.array_equal(out[ay : ay + gh, ax : ax + gw], img[ay : ay + gh, ax : ax + gw])
        assert np.array_equal(out[:200, :200], img[:200, :200])  # far corner untouched

    def test_detected_inpaint_lowers_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "preferred_inpaint_backend", lambda: "cv2")
        mark = registry.get_mark("doubao")
        img, _ = _compose_textmark(DoubaoEngine())
        before = mark.detect(img)
        assert before.detected  # the composed mark is detectable
        out, region = mark.remove(img, method="inpaint")
        assert region is not None
        assert mark.detect(out).confidence < before.confidence

    def test_reverse_alpha_method_still_selectable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "inpaint_model_available", lambda: True)  # would be inpaint on auto
        img, _ = _compose_textmark(DoubaoEngine())
        # explicit reverse-alpha bypasses the inpaint fallback even with a model present
        out, region = registry.get_mark("doubao").remove(img, method="reverse-alpha")
        assert region is not None
        assert not np.array_equal(out, img)


class TestBackendSelection:
    """MI-GAN is the preferred inpaint backend (light, droplet-friendly); big-LaMa
    is NOT auto-selected. cv2 is the floor when no ONNX model is present."""

    def test_prefers_migan_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from remove_ai_watermarks import region_eraser

        monkeypatch.setattr(region_eraser, "migan_available", lambda: True)
        assert registry.preferred_inpaint_backend() == "migan"

    def test_cv2_when_no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from remove_ai_watermarks import region_eraser

        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        assert registry.preferred_inpaint_backend() == "cv2"

    def test_inpaint_model_available_reflects_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from remove_ai_watermarks import region_eraser

        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        assert not registry.inpaint_model_available()
        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        assert registry.inpaint_model_available()

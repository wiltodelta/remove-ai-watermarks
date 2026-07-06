"""Jimeng-basic 'AI生成' pill: capture-less mark (detect via synthetic silhouette
edge-NCC, remove via inpaint). No model download -- cv2 fallback / pure logic only."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from remove_ai_watermarks import watermark_registry as registry
from remove_ai_watermarks.pill_engine import _DETECT_THRESHOLD, PillEngine

_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


def _font_ok() -> bool:
    try:
        ImageFont.truetype(_FONT, 20)
        return True
    except Exception:
        return False


_HAS_FONT = _font_ok()
_needs_font = pytest.mark.skipif(
    not _HAS_FONT, reason="CJK font unavailable (compose helper needs it; asset is committed)"
)


def _compose_pill(w: int = 1200, h: int = 1600, bg: int = 150) -> np.ndarray:
    """Composite a semi-transparent 'AI生成' pill top-left onto a flat BGR frame."""
    img = Image.new("RGB", (w, h), (bg, bg, bg))
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    mw, mh = int(0.167 * w), int(0.09 * w)
    mx, my = int(0.03 * w), int(0.02 * w)
    d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=mh // 3, outline=(255, 255, 255, 150), width=3)
    font = ImageFont.truetype(_FONT, int(mh * 0.5))
    d.text((mx + mw // 6, my + mh // 5), "AI生成", font=font, fill=(255, 255, 255, 170))
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    return np.asarray(out)[:, :, ::-1].copy()  # RGB->BGR


class TestPillDetect:
    @_needs_font
    def test_detects_composited_pill(self) -> None:
        det = PillEngine().detect(_compose_pill())
        assert det.detected
        assert det.confidence >= _DETECT_THRESHOLD

    def test_clean_frame_does_not_fire(self) -> None:
        clean = np.full((1600, 1200, 3), 150, np.uint8)
        assert not PillEngine().detect(clean).detected

    def test_small_image_no_fire(self) -> None:
        assert not PillEngine().detect(np.full((40, 40, 3), 150, np.uint8)).detected


class TestPillMask:
    def test_footprint_mask_top_left_geometry(self) -> None:
        mask = PillEngine().footprint_mask(np.full((1600, 1200, 3), 150, np.uint8))
        assert mask is not None
        assert mask.shape == (1600, 1200)
        assert mask.any()
        ys, xs = np.where(mask > 0)
        # pill sits top-left: mask mass in the top-left quadrant
        assert ys.mean() < 800
        assert xs.mean() < 600


class TestPillRegistry:
    def test_pill_is_capture_less(self) -> None:
        m = registry.get_mark("jimeng_pill")
        assert m.has_capture is False

    def test_capture_less_routes_every_method_to_inpaint(self) -> None:
        # a capture-less mark cannot reverse-alpha; even explicit reverse-alpha -> inpaint
        assert registry.resolve_removal_method("reverse-alpha", False) == "inpaint"
        assert registry.resolve_removal_method("auto", False) == "inpaint"
        assert registry.resolve_removal_method("inpaint", False) == "inpaint"


class TestPillGate:
    """The pill is kept only when the image is CONFIRMED Jimeng (TC260 metadata OR
    the bottom-right wordmark fired) and NOT Doubao. Fakes detect_marks so no image
    content is needed; cv2 backend so nothing downloads."""

    @staticmethod
    def _fakes(monkeypatch: pytest.MonkeyPatch, keys: set[str]) -> None:
        from remove_ai_watermarks.watermark_registry import MarkDetection

        labels = {
            "doubao": "Doubao 豆包AI生成 text",
            "jimeng": "Jimeng 即梦AI wordmark",
            "jimeng_pill": "Jimeng AI生成 pill",
        }
        monkeypatch.setattr(registry, "preferred_inpaint_backend", lambda: "cv2")
        monkeypatch.setattr(
            registry,
            "detect_marks",
            lambda image, *, include_explicit=True: [
                MarkDetection(k, labels[k], "loc", True, 0.6, (10, 10, 40, 40)) for k in keys
            ],
        )

    def test_pill_kept_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), pill_metadata=True)
        assert "Jimeng AI生成 pill" in removed

    def test_pill_kept_via_wordmark_without_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # metadata-stripped upload: the bottom-right wordmark confirms Jimeng
        self._fakes(monkeypatch, {"jimeng", "jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), pill_metadata=False)
        assert "Jimeng AI生成 pill" in removed

    def test_pill_dropped_without_metadata_or_wordmark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), pill_metadata=False)
        assert "Jimeng AI生成 pill" not in removed

    def test_pill_dropped_on_doubao_even_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fakes(monkeypatch, {"doubao", "jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), pill_metadata=True)
        assert "Doubao 豆包AI生成 text" in removed
        assert "Jimeng AI生成 pill" not in removed

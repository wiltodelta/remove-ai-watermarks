"""Jimeng-basic 'AI生成' pill: synthetic-silhouette edge-NCC and inpainting.

No model download -- cv2 fallback / pure logic only. The separate visible-example
suite exercises the detector against a published real-world carrier.
"""

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

    def test_corroborated_provenance_relaxes_borderline_pill_only_when_confirmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eng = PillEngine(include_label=False)
        monkeypatch.setattr(eng, "_match", lambda _img: (0.20, (20, 20, 160, 80)))
        frame = np.full((800, 600, 3), 140, np.uint8)
        strict, relaxed = eng.detect_both(frame)
        assert not strict.detected
        assert relaxed.detected
        assert not eng.detect(frame, provenance=False).detected
        assert eng.detect(frame, provenance=True).detected


def _textured_frame(w: int = 300, h: int = 400, bg: int = 150) -> np.ndarray:
    """Flat frame with a high-frequency checkerboard over the top-left footprint,
    so the pill footprint reads as TEXTURED (an inpaint there would smear)."""
    img = np.full((h, w, 3), bg, np.uint8)
    fx, fy, fw, fh = int(0.012 * w), int(0.006 * h), int(0.205 * w), int(0.115 * w)
    yy, xx = np.mgrid[0:fh, 0:fw]
    checker = (((xx // 3) + (yy // 3)) % 2 * 255).astype(np.uint8)
    img[fy : fy + fh, fx : fx + fw] = checker[:, :, None]
    return img


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


class TestFootprintFlatness:
    """The metadata-only pill arm removes only on a flat footprint (safe inpaint)."""

    def test_flat_frame_is_flat(self) -> None:
        assert PillEngine().footprint_is_flat(np.full((1600, 1200, 3), 150, np.uint8))

    def test_textured_frame_is_not_flat(self) -> None:
        eng = PillEngine()
        assert not eng.footprint_is_flat(_textured_frame(1200, 1600))
        # median-Sobel texture is well above the flat threshold on the checkerboard
        assert eng.footprint_texture(_textured_frame(1200, 1600)) > 6.0


class TestPillRegistry:
    def test_pill_registered_top_left(self) -> None:
        m = registry.get_mark("jimeng_pill")
        assert m.location == "top-left"
        assert m.in_auto is True

    def test_pill_mask_is_top_left_via_registry(self) -> None:
        # The registry mask callable delegates to the pill engine's top-left footprint.
        mask = registry.get_mark("jimeng_pill")._mask(np.full((1600, 1200, 3), 150, np.uint8))
        assert mask is not None
        assert mask.any()


class TestPillGate:
    """Pill removal is gated (``_keep_pill``): the reliable bottom-right wordmark
    removes it unrestricted, the metadata arm (``"jimeng"`` provenance) removes it ONLY
    on a flat footprint (safe fill), Doubao/no-confirmation never remove it. Fakes each
    mark's detect so no image content is needed; cv2 backend so nothing downloads. Frame
    flatness matters, so tests pass a flat or textured frame."""

    @staticmethod
    def _fakes(monkeypatch: pytest.MonkeyPatch, keys: set[str]) -> list[str]:
        """Fake every mark's verdict. Returns the list the fake appends each call to.

        BOTH entry points must be faked: the arbiter's perception pass goes through
        ``detect_both`` (one scan, two verdicts) while the removal path still calls
        ``detect``. Patching only one leaves the real detectors running on a synthetic
        frame, where they find nothing -- so a "dropped" assertion would pass for
        entirely the wrong reason. The returned call log is what pins that.
        """
        from remove_ai_watermarks.watermark_registry import KnownMark, MarkDetection

        labels = {
            "doubao": "Doubao 豆包AI生成 text",
            "jimeng": "Jimeng 即梦AI wordmark",
            "jimeng_pill": "Jimeng AI生成 pill",
        }
        monkeypatch.setattr(registry, "preferred_inpaint_backend", lambda: "cv2")
        calls: list[str] = []

        def fake_detect(self: KnownMark, image: object, *, provenance: bool = False) -> MarkDetection:
            calls.append(self.key)
            return MarkDetection(
                self.key, labels.get(self.key, self.key), "loc", self.key in keys, 0.6, (10, 10, 40, 40)
            )

        def fake_detect_both(self: KnownMark, image: object) -> tuple[MarkDetection, MarkDetection]:
            d = fake_detect(self, image)
            return d, d

        monkeypatch.setattr(registry.KnownMark, "detect", fake_detect)
        monkeypatch.setattr(registry.KnownMark, "detect_both", fake_detect_both)
        return calls

    def test_pill_kept_with_metadata_on_flat_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # jimeng provenance (TC260) + flat background -> safe fill, remove
        self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), provenance=frozenset({"jimeng"}))
        assert "Jimeng AI生成 pill" in removed

    def test_pill_dropped_with_metadata_on_textured_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # jimeng provenance + textured background (ceiling-like) -> fill would smear, skip
        calls = self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(_textured_frame(), provenance=frozenset({"jimeng"}))
        assert "jimeng_pill" in calls, "the fake detector never ran -- this would pass vacuously"
        assert "Jimeng AI生成 pill" not in removed

    def test_pill_kept_via_wordmark_ignores_texture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # wordmark confirmation (~94% precise, survives metadata stripping) is NOT
        # texture-gated: a wordmark-confirmed pill is removed even on a textured frame
        self._fakes(monkeypatch, {"jimeng", "jimeng_pill"})
        _, removed = registry.remove_auto_marks(_textured_frame())
        assert "Jimeng AI生成 pill" in removed

    def test_pill_dropped_on_textured_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The metadata arm keeps the flatness guard: textured false fires visibly smear.
        calls = self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(_textured_frame(), provenance=frozenset({"jimeng"}))
        assert "jimeng_pill" in calls, "the fake detector never ran -- this would pass vacuously"
        assert "Jimeng AI生成 pill" not in removed

    def test_pill_dropped_without_metadata_or_wordmark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fakes(monkeypatch, {"jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8))
        assert "Jimeng AI生成 pill" not in removed

    def test_pill_dropped_on_doubao_even_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # doubao is faked as detected, which drives the pill gate (pill never rides on a
        # Doubao detection). The same flat + jimeng-metadata setup WITHOUT doubao keeps the
        # pill (test_pill_kept_with_metadata_on_flat_footprint), so doubao is the
        # differentiator. Doubao itself is not asserted in `removed` here: this synthetic
        # frame is flat with no real glyph, so the text mask has nothing to fill (its real
        # removal is covered by TestRealSample on the committed doubao sample).
        self._fakes(monkeypatch, {"doubao", "jimeng_pill"})
        _, removed = registry.remove_auto_marks(np.full((400, 300, 3), 150, np.uint8), provenance=frozenset({"jimeng"}))
        assert "Jimeng AI生成 pill" not in removed


def test_detect_bgra_no_crash() -> None:
    # A 4-channel BGRA array must be normalized, not crash cv2.cvtColor(BGR2GRAY) (#10).
    bgra = np.zeros((256, 256, 4), np.uint8)
    det = PillEngine().detect(bgra)
    assert det.detected in (True, False)
    assert PillEngine().footprint_texture(bgra) >= 0.0

"""Tests for the known-visible-watermark registry (localize -> fill)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as reg

DOUBAO_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


class TestCatalog:
    def test_keys(self):
        assert reg.mark_keys() == ["gemini", "doubao", "jimeng", "qwen", "samsung", "jimeng_pill"]

    def test_all_in_auto(self):
        assert all(m.in_auto for m in reg.known_marks())

    def test_marks_expose_detect_and_mask(self):
        # Every mark drives the uniform localize -> fill contract: a detect callable
        # (verdict + bbox, no mask) and a mask callable (full-frame footprint).
        for m in reg.known_marks():
            assert callable(m._detect)
            assert callable(m._mask)

    def test_locations(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].location == "bottom-right"
        assert by_key["doubao"].location == "bottom-right"
        assert by_key["jimeng"].location == "bottom-right"
        assert by_key["samsung"].location == "bottom-left"
        assert by_key["jimeng_pill"].location == "top-left"

    def test_get_mark_unknown_raises(self):
        with pytest.raises(KeyError):
            reg.get_mark("nope")


class TestScan:
    def test_detect_marks_scans_all(self):
        img = np.zeros((256, 256, 3), np.uint8)
        keys = {d.key for d in reg.detect_marks(img)}
        assert keys == {"gemini", "doubao", "jimeng", "qwen", "samsung", "jimeng_pill"}

    def test_blank_image_no_auto_mark(self):
        dets = reg.detect_marks(np.zeros((256, 256, 3), np.uint8), include_explicit=False)
        assert not any(d.detected for d in dets)

    @pytest.mark.parametrize("shape", [(1, 1, 3), (8, 8, 3), (15, 15, 3), (12, 300, 3), (300, 10, 3)])
    def test_tiny_image_no_crash(self, shape):
        """Regression: an image whose short side is < 16 px (below the Gemini template
        floor) must yield no detection, not crash. detect_marks/remove_auto_marks are
        the public visible/all/batch path; a tiny thumbnail in a batch used to take the
        whole auto pass down with an IndexError (empty candidate list dereference)."""
        img = np.full(shape, 100, np.uint8)
        assert not any(d.detected for d in reg.detect_marks(img, include_explicit=False))
        result, removed = reg.remove_auto_marks(img, backend="cv2")
        assert removed == []
        assert result.shape == img.shape

    @pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 5, 4), (0, 0)])
    def test_forced_remove_on_empty_array_no_crash(self, shape):
        """Regression: footprint_mask ran to_bgr (cvtColor) before any size check, so a
        forced remove on a zero-size ndarray crashed (cv2.error on an empty Mat). detect
        already guarded this; footprint_mask must too. Covers the text + gemini engines."""
        empty = np.zeros(shape, np.uint8)
        for key in ("doubao", "jimeng", "qwen", "samsung", "gemini"):
            _result, mask = reg.get_mark(key).remove(empty, force=True)
            assert mask is None


class TestBackendResolution:
    def test_auto_resolves_to_available_backend(self):
        assert reg.resolve_backend("auto") in ("cv2", "migan", "lama")

    def test_explicit_backend_passes_through(self):
        assert reg.resolve_backend("cv2") == "cv2"
        assert reg.resolve_backend("lama") == "lama"

    def test_cv2_fallback_warns_once(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        import logging

        from remove_ai_watermarks import region_eraser

        monkeypatch.setattr(region_eraser, "lama_available", lambda: False)
        monkeypatch.setattr(region_eraser, "migan_available", lambda: False)
        monkeypatch.setattr(reg, "_warned_cv2_fallback", False)
        with caplog.at_level(logging.WARNING):
            assert reg.preferred_inpaint_backend() == "cv2"
            assert reg.preferred_inpaint_backend() == "cv2"
        assert sum("cv2 classical inpaint" in r.message for r in caplog.records) == 1


class TestFill:
    def test_fill_erases_masked_region(self):
        # A bright square on a flat field, masked, is inpainted away (cv2 backend).
        img = np.full((128, 128, 3), 60, np.uint8)
        img[40:70, 40:70] = 240
        mask = np.zeros((128, 128), np.uint8)
        mask[36:74, 36:74] = 255
        out = reg.fill(img, mask, backend="cv2")
        assert out.shape == img.shape
        # the masked bright square is pulled toward the surrounding field
        assert int(out[55, 55].mean()) < 160

    def test_fill_empty_mask_is_noop(self):
        img = np.full((64, 64, 3), 100, np.uint8)
        out = reg.fill(img, np.zeros((64, 64), np.uint8), backend="cv2")
        assert np.array_equal(out, img)


class TestProvenanceGate:
    """The Gemini trust gate relaxes from GEMINI_SPARKLE_TRUST_CONF to
    _GEMINI_PROVENANCE_MIN_CONF when provenance confirms Google; tested
    deterministically by stubbing the engine's raw detection confidence."""

    def _stub(self, monkeypatch: pytest.MonkeyPatch, conf: float) -> None:
        from remove_ai_watermarks.gemini_engine import DetectionResult

        # `detected` mirrors the engine's own internal floor (0.35), which is
        # independent of the registry gate under test here.
        def fake_detect(image, force_size=None, *, trust_provenance=False):
            return DetectionResult(detected=conf >= 0.35, confidence=conf, region=(10, 10, 48, 48))

        monkeypatch.setattr(reg._engine("gemini"), "detect_watermark", fake_detect)

    def test_midband_conf_needs_provenance(self, monkeypatch: pytest.MonkeyPatch):
        # Comfortably inside the relaxed band: demoted without provenance, trusted with it.
        conf = (reg._GEMINI_PROVENANCE_MIN_CONF + reg.GEMINI_SPARKLE_TRUST_CONF) / 2
        self._stub(monkeypatch, conf)
        img = np.zeros((256, 256, 3), np.uint8)
        assert reg.get_mark("gemini").detect(img).detected is False
        assert reg.get_mark("gemini").detect(img, provenance=True).detected is True

    def test_below_provenance_gate_rejected_even_with_provenance(self, monkeypatch: pytest.MonkeyPatch):
        """Provenance relaxes the gate, it does not remove it.

        Guards the 2026-07-18 raise of _GEMINI_PROVENANCE_MIN_CONF (0.35 -> 0.42).
        The engine still reports `detected` down at its own 0.35 floor, so without
        the registry gate this confidence would be accepted and inpainted. Measured
        precision just below the gate was 13% (n=30, 95% CI 5-30%) on real
        Google-metadata uploads -- i.e. ~7 of 8 accepts there destroy pixels on an
        image that never carried a sparkle, and report a removal that did not happen.
        """
        # 0.38 is inside the measured 13%-precision band and above the engine's own
        # 0.35 floor, so the engine reports `detected` and only the registry gate can
        # reject it. Hardcoded on purpose: if the gate is ever lowered back under this
        # value, this test must fail on the BEHAVIOUR below, not on its own arithmetic.
        self._stub(monkeypatch, 0.38)
        img = np.zeros((256, 256, 3), np.uint8)
        assert reg.get_mark("gemini").detect(img).detected is False
        assert reg.get_mark("gemini").detect(img, provenance=True).detected is False

    def test_provenance_gate_stays_below_the_strict_gate(self):
        """The relaxed gate must actually relax, and must not collapse onto the floor."""
        assert 0.35 < reg._GEMINI_PROVENANCE_MIN_CONF < reg.GEMINI_SPARKLE_TRUST_CONF

    def test_high_conf_detected_either_way(self, monkeypatch: pytest.MonkeyPatch):
        self._stub(monkeypatch, 0.72)
        img = np.zeros((256, 256, 3), np.uint8)
        assert reg.get_mark("gemini").detect(img).detected is True
        assert reg.get_mark("gemini").detect(img, provenance=True).detected is True


@pytest.mark.skipif(not DOUBAO_SAMPLE.exists(), reason="doubao sample not present")
class TestRealSample:
    def test_doubao_sample_detected(self):
        from remove_ai_watermarks.image_io import imread

        fired = [d.key for d in reg.detect_marks(imread(DOUBAO_SAMPLE), include_explicit=False) if d.detected]
        assert "doubao" in fired

    def test_doubao_remove_returns_region(self):
        from remove_ai_watermarks.image_io import imread

        img = imread(DOUBAO_SAMPLE)
        result, region = reg.get_mark("doubao").remove(img, backend="cv2")
        assert region is not None
        assert result.shape == img.shape


class TestLocalizeFill:
    def test_clean_corner_is_untouched(self):
        # No glyph in the corner -> no mask -> remove is a no-op copy.
        img = np.zeros((512, 512, 3), np.uint8)
        result, region = reg.get_mark("doubao").remove(img, backend="cv2")
        assert region is None
        assert np.array_equal(result, img)


class TestSensitivity:
    """``resolve_trust`` turns the sensitivity policy + evidence into the per-mark
    trust level the engines consume."""

    def test_strict_never_relaxes(self):
        # even with metadata provenance, strict keeps the conservative gate
        assert (
            reg.resolve_trust("gemini", sensitivity="strict", provenance=frozenset({"gemini"}), strict_keys=set())
            == "strict"
        )

    def test_auto_relaxes_on_own_metadata(self):
        assert (
            reg.resolve_trust("gemini", sensitivity="auto", provenance=frozenset({"gemini"}), strict_keys=set())
            == "confirmed"
        )

    def test_auto_strict_without_evidence(self):
        assert reg.resolve_trust("gemini", sensitivity="auto", provenance=frozenset(), strict_keys=set()) == "strict"

    def test_auto_cross_mark_same_product(self):
        # a detected Jimeng wordmark relaxes the Jimeng pill (same product, other corner)
        assert (
            reg.resolve_trust("jimeng_pill", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"})
            == "confirmed"
        )

    def test_auto_no_cross_mark_across_products(self):
        # a detected Jimeng wordmark must NOT relax Doubao (distinct products, same corner)
        assert (
            reg.resolve_trust("doubao", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"}) == "strict"
        )

    def test_remove_auto_marks_accepts_all_sensitivities(self):
        blank = np.zeros((256, 256, 3), np.uint8)
        for s in ("auto", "strict"):
            _, removed = reg.remove_auto_marks(blank, sensitivity=s, backend="cv2")
            assert removed == []


class TestNoBlanketRelaxation:
    """There is NO path that relaxes a mark's gate without same-product evidence.

    ``assume_ai`` was that path and was removed 2026-07-19: it bypassed every mark's
    false-positive gate on the caller's bare assertion that the image is AI, which says
    nothing about WHICH vendor or WHERE -- exactly what the bypass is contracted to
    require. Before it carried a confidence floor it filled a phantom sparkle on 59.8%
    of genuine camera photos. A user who can SEE a mark is served by `erase --region`
    (they supply the coordinates) or `--mark <name> --no-detect` for a text mark.
    """

    def test_sensitivity_has_exactly_two_levels(self):
        import typing

        assert set(typing.get_args(reg.Sensitivity)) == {"auto", "strict"}

    def test_trust_ladder_has_no_assumed_level(self):
        import typing

        assert set(typing.get_args(reg.Trust)) == {"strict", "confirmed"}

    def test_no_sensitivity_relaxes_without_same_product_evidence(self):
        for sens in ("auto", "strict"):
            assert reg.resolve_trust("gemini", sensitivity=sens, provenance=frozenset(), strict_keys=set()) == "strict"

    def test_the_assumed_floor_helper_is_gone(self):
        """It existed only to make the blanket relaxation tolerable."""
        assert not hasattr(reg, "assumed_floor_ok")
        assert not hasattr(reg, "_ASSUMED_CONF_FLOOR")

    def test_the_removed_value_raises_instead_of_silently_meaning_auto(self):
        """`Sensitivity` is a Literal and unenforced at runtime, so a 0.15 caller passing
        the removed value would quietly get `auto` -- a silent semantic change on exactly
        the release where they need to be told. The error names the replacement."""
        import pytest

        with pytest.raises(ValueError, match="erase"):
            reg.validate_sensitivity("assume_ai")
        with pytest.raises(ValueError, match="unknown sensitivity"):
            reg.validate_sensitivity("aggressive")
        assert reg.validate_sensitivity("auto") == "auto"
        assert reg.validate_sensitivity("strict") == "strict"

    def test_context_rejects_it_too(self):
        """The arbiter's own entry point validates, so a direct `decide()` caller cannot
        smuggle the removed mode past the public API."""
        import pytest

        with pytest.raises(ValueError, match=r"removed in 0\.16"):
            reg.Context(sensitivity="assume_ai")


class TestArbiter:
    """``decide`` is the PURE removal arbiter: (candidates, context) -> ordered
    winners, no image / no I/O. Tested in isolation by handing it fabricated
    Candidates -- this is the payoff of separating decision from perception."""

    @staticmethod
    def _c(key, *, strict=False, relaxed=False, flat=False):
        feats = {"footprint_flat": 1.0} if flat else {}
        return reg.Candidate(key, f"L:{key}", strict, relaxed, feats)

    def _keys(self, cands, ctx):
        return {d.candidate.key for d in reg.decide(cands, ctx)}

    def test_empty(self):
        assert reg.decide([], reg.Context()) == []

    def test_strict_uses_strict_verdict(self):
        # relaxed-only detection must NOT fire under strict
        assert self._keys([self._c("gemini", relaxed=True)], reg.Context(sensitivity="strict")) == set()

    def test_auto_relaxes_on_provenance(self):
        c = [self._c("gemini", relaxed=True)]
        assert self._keys(c, reg.Context(provenance=frozenset({"gemini"}))) == {"gemini"}
        assert self._keys(c, reg.Context()) == set()  # no evidence -> strict verdict (not fired)

    def test_cross_mark_relaxes_pill_via_jimeng(self):
        cands = [self._c("jimeng", strict=True, relaxed=True), self._c("jimeng_pill", relaxed=True, flat=True)]
        assert self._keys(cands, reg.Context()) == {"jimeng", "jimeng_pill"}

    def test_pill_dropped_on_doubao(self):
        cands = [
            self._c("doubao", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=True),
        ]
        keys = self._keys(cands, reg.Context(provenance=frozenset({"jimeng"})))
        assert "doubao" in keys
        assert "jimeng_pill" not in keys

    def test_pill_dropped_on_qwen(self):
        # A Qwen frame is TC260 too but is not Jimeng-basic either: a confident
        # bottom-right 千问AI生成 detection suppresses the pill exactly like Doubao's.
        cands = [
            self._c("qwen", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=True),
        ]
        keys = self._keys(cands, reg.Context(provenance=frozenset({"jimeng"})))
        assert "qwen" in keys
        assert "jimeng_pill" not in keys

    def test_pill_metadata_arm_gated_on_flatness(self):
        ctx = reg.Context(provenance=frozenset({"jimeng"}))
        assert self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=True)], ctx) == {"jimeng_pill"}
        assert self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=False)], ctx) == set()

    def test_pill_wordmark_arm_ignores_flatness(self):
        # wordmark present -> pill removed even on a textured (non-flat) footprint
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
        ]
        assert "jimeng_pill" in self._keys(cands, reg.Context())

    def test_weak_pill_detection_does_not_confirm_the_jimeng_wordmark(self):
        """The pill is too false-fire-prone (~7%) to grant a sibling `confirmed` trust.

        Corpus-measured defect (2026-07-18): a pill false fire on clean non-ByteDance
        content confirmed jimeng, relaxing its NCC gate 0.45 -> 0.3825; jimeng then
        false-fired, and _keep_pill's wordmark arm removed the pill UNRESTRICTED,
        skipping the flatness guard. Closed loop on the default `auto` path.
        """
        # Only the pill is strictly detected. jimeng scores in the band that is
        # reachable ONLY via the relaxed gate.
        cands = [
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
            self._c("jimeng", strict=False, relaxed=True),
        ]
        fired = {d.candidate.key for d in reg.decide(cands, reg.Context("auto", frozenset()))}
        assert "jimeng" not in fired, "a weak pill hit must not relax the jimeng wordmark"
        # and with jimeng gone, the pill loses the wordmark arm too -- no unrestricted
        # removal of a textured footprint on content nothing confirmed.
        assert "jimeng_pill" not in fired

    def test_real_jimeng_wordmark_still_corroborates_the_pill(self):
        """The fix removes only the pill's TESTIMONY, not the wordmark's."""
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
        ]
        fired = {d.candidate.key for d in reg.decide(cands, reg.Context("auto", frozenset()))}
        assert fired == {"jimeng", "jimeng_pill"}


class TestProvenanceMaskThreading:
    """Regression for the provenance-relaxed Gemini no-op (#1) and the false 'removed'
    label (#2). Before the fix, footprint_mask re-detected WITHOUT trust_provenance, the
    FP gate demoted the sparkle to detected=False, the mask came back None, yet
    remove_auto_marks still reported the mark as removed."""

    def test_relaxed_sparkle_yields_mask(self, monkeypatch: pytest.MonkeyPatch):
        # A sparkle a strict re-detect would demote (detected False) but a
        # provenance-relaxed detect accepts must still produce a removal mask.
        from remove_ai_watermarks.gemini_engine import DetectionResult

        def fake(image, force_size=None, *, trust_provenance=False):
            return DetectionResult(
                detected=trust_provenance, confidence=0.42 if trust_provenance else 0.30, region=(400, 400, 60)
            )

        monkeypatch.setattr(reg._engine("gemini"), "detect_watermark", fake)
        img = np.full((512, 512, 3), 90, np.uint8)
        assert reg.get_mark("gemini").localize(img, provenance=True).mask is not None
        assert reg.get_mark("gemini").localize(img, provenance=False).mask is None

    def test_no_label_when_mask_none(self, monkeypatch: pytest.MonkeyPatch):
        # A decided mark whose mask comes back None must NOT be reported as removed.
        from remove_ai_watermarks.gemini_engine import DetectionResult

        eng = reg._engine("gemini")
        monkeypatch.setattr(
            eng,
            "detect_watermark",
            lambda image, force_size=None, *, trust_provenance=False: DetectionResult(True, 0.9, (10, 10, 40)),
        )
        monkeypatch.setattr(eng, "footprint_mask", lambda image, *, force=False, region=None, dilate=None: None)
        _, removed = reg.remove_auto_marks(np.zeros((256, 256, 3), np.uint8), sensitivity="strict", backend="cv2")
        assert "Google Gemini sparkle" not in removed

"""Policy-level tests for the shared text-mark engine config.

These assert calibrated TUNING, not algorithm behavior --
they exist so a future edit cannot silently revert a calibrated constant back to a
value that was measured to be wrong. The measurements themselves live in
`docs/module-internals.md` and in the comment at
`_text_mark_engine._DEFAULT_PROVENANCE_NCC_FACTOR`.
"""


class TestRivalMargin:
    """Detection among same-corner marks is COMPETITIVE, not just absolute.

    Doubao "豆包AI生成" and Jimeng "★ 即梦AI" both sit bottom-right in near-white CJK
    and survive binarization as similar blobs, so an absolute NCC gate cannot tell
    them apart because many Jimeng false additions were Doubao marks. Calibration
    showed that the relative template margin separates them without reducing genuine
    Jimeng detections.
    """

    def test_jimeng_competes_against_doubao(self):
        from remove_ai_watermarks import jimeng_engine

        assert "doubao_alpha.png" in jimeng_engine._CONFIG.rivals

    def test_doubao_has_no_rival_margin(self):
        """Asymmetric by measurement, not oversight: the symmetric gate cost Doubao 7
        genuine detections to prevent 5 false ones (1.4:1 against), while Jimeng gained
        25pp for free. Doubao's absolute detector is already 86% precise."""
        from remove_ai_watermarks import doubao_engine

        assert doubao_engine._CONFIG.rivals == ()

    def test_a_doubao_shaped_blob_loses_the_jimeng_margin(self):
        """The decisive case: a blob matching Doubao BETTER than Jimeng must not be
        booked as a Jimeng wordmark, however high its absolute Jimeng score."""
        import numpy as np

        from remove_ai_watermarks import jimeng_engine
        from remove_ai_watermarks._text_mark_engine import glyph_silhouette

        eng = jimeng_engine.JimengEngine()
        doubao_blob = glyph_silhouette("doubao_alpha.png")
        assert doubao_blob is not None
        canvas = np.zeros((doubao_blob.shape[0] + 20, doubao_blob.shape[1] + 20), np.uint8)
        canvas[10 : 10 + doubao_blob.shape[0], 10 : 10 + doubao_blob.shape[1]] = doubao_blob
        width = int(doubao_blob.shape[1] / jimeng_engine._CONFIG.alpha_width_frac)
        jimeng_score = eng._template_match_score(canvas, width)
        assert not eng._rival_margin_ok(jimeng_score, canvas, width)


class TestPerMarkProvenanceRelaxation:
    """The provenance NCC relaxation is PER MARK, not one shared multiplier.

    Calibration showed that one shared relaxation factor was too permissive for
    Jimeng because its relaxed silhouette confuses other bottom-right text marks.
    """

    def test_samsung_provenance_floor_stays_at_040_and_korean_stays_strict(self):
        """Base Samsung lifts strict detection to 0.50 against corner captions while keeping
        its samsung_genai-confirmed floor at 0.40 (never 0.28, which admits clean texture);
        the Korean variant stays strict-only at 0.46."""
        import pytest

        from remove_ai_watermarks import samsung_engine

        cfg = samsung_engine._CONFIG
        assert cfg.detect_ncc_threshold == 0.50
        assert cfg.detect_ncc_threshold * cfg.provenance_ncc_factor == pytest.approx(0.40)
        ko_cfg = samsung_engine.SamsungEngine()._variants[0].config
        assert ko_cfg.provenance_ncc_factor == 1.0


class TestScaleBasis:
    """Mark geometry scales with a PER-MARK image dimension, measured not assumed.

    Every tuned fraction was calibrated on PORTRAIT captures, where width and short
    side coincide, so the basis was never exercised until landscape inputs were
    measured. Calibration showed that a width-scaled Doubao box is inflated by the
    aspect ratio on a wide image and can miss the glyph. A short-side basis recovered
    the affected landscape cases. The same switch broke Jimeng, whose wordmark tracks
    the width, hence per-mark rather than a house rule.
    """

    def test_doubao_scales_with_the_short_side(self):
        from remove_ai_watermarks import doubao_engine

        assert doubao_engine._CONFIG.scale_basis == "short"

    def test_jimeng_scales_with_width(self):
        """Measured, not an oversight: the short-side basis took jimeng's labeled
        landscape positives from 13/13 to 0/13."""
        from remove_ai_watermarks import jimeng_engine

        assert jimeng_engine._CONFIG.scale_basis == "width"

    def test_samsung_keeps_width_because_it_is_unmeasured(self):
        """There is no calibration evidence for changing this basis."""
        from remove_ai_watermarks import samsung_engine

        assert samsung_engine._CONFIG.scale_basis == "width"

    def test_basis_only_differs_on_non_square_images(self):
        """The basis is a no-op wherever width IS the short side, which is why the bug
        survived: every calibration capture was portrait."""
        import numpy as np

        from remove_ai_watermarks import doubao_engine, jimeng_engine

        portrait = np.zeros((1600, 900, 3), np.uint8)
        landscape = np.zeros((900, 1600, 3), np.uint8)
        d, j = doubao_engine.DoubaoEngine(), jimeng_engine.JimengEngine()
        assert d.scale_base(portrait) == j.scale_base(portrait) == 900
        assert d.scale_base(landscape) == 900
        assert j.scale_base(landscape) == 1600

    def test_landscape_box_stays_inside_the_frame(self):
        """The concrete failure: a width-scaled box on a wide image overshoots the
        mark's real footprint. The short-side box must be proportionally smaller."""
        import numpy as np

        from remove_ai_watermarks import doubao_engine

        landscape = np.zeros((900, 2400, 3), np.uint8)
        loc = doubao_engine.DoubaoEngine().locate(landscape)
        assert loc.w < int(2400 * doubao_engine._CONFIG.width_frac)
        assert loc.x + loc.w <= 2400
        assert loc.y + loc.h <= 900


class TestTophatFrontend:
    """Detection can correlate the CONTINUOUS top-hat instead of a binarized blob.

    `extract_mask` thresholds the top-hat into a 0/255 glyph blob, which is fine for a
    mark stamped bold and opaque and destructive for a faint one -- a thin translucent
    overlay shatters into specks and no template can match a blob that is not there
    (measured: 千问 scored 0.170 mean vs doubao's 0.723 through the binary path, 0% over
    the gate). The `tophat` front-end never binarizes: the saturation/luma gates become
    weights, and the response is max-normalized so the score is contrast-invariant.

    Calibration showed improved Doubao recall without reducing precision.
    """

    def test_doubao_uses_the_continuous_frontend(self):
        from remove_ai_watermarks import doubao_engine

        assert doubao_engine._CONFIG.detect_frontend == "tophat"

    def test_samsung_uses_the_continuous_frontend_after_measurement(self):
        """Samsung's own real-positive and clean-control study supports top-hat."""
        from remove_ai_watermarks import samsung_engine

        assert samsung_engine._CONFIG.detect_frontend == "tophat"
        assert samsung_engine._CONFIG.ladder == (1.0,)

    def test_jimeng_stays_binary_until_measured(self):
        """A front-end switch must be measured per mark before it ships."""
        from remove_ai_watermarks import jimeng_engine

        assert jimeng_engine._CONFIG.detect_frontend == "binary"

    def test_response_is_contrast_invariant(self):
        """The whole point: a faint mark and a bold one produce the same response, so a
        single threshold works for both. Binarizing is what loses the faint one."""
        import numpy as np

        from remove_ai_watermarks import doubao_engine

        eng = doubao_engine.DoubaoEngine()
        h, w = 400, 900
        out = []
        for amplitude in (12, 90):  # a barely-there overlay and a bold one
            img = np.full((h, w, 3), 100, np.uint8)
            loc = eng.locate(img)
            x, y, bw, bh = loc.bbox
            img[y + bh // 3 : y + 2 * bh // 3, x + bw // 4 : x + 3 * bw // 4] = 100 + amplitude
            resp = eng.tophat_response(img, loc)
            assert resp is not None
            out.append(resp)
        # max-normalized, so the two responses agree despite a 7.5x contrast difference
        assert abs(int(out[0].max()) - int(out[1].max())) <= 1

    def test_flat_input_yields_no_response(self):
        """A blank corner has no top-hat at all; the engine must return None rather
        than divide by a zero peak."""
        import numpy as np

        from remove_ai_watermarks import doubao_engine

        eng = doubao_engine.DoubaoEngine()
        img = np.full((400, 900, 3), 128, np.uint8)
        assert eng.tophat_response(img, eng.locate(img)) is None

    def test_threshold_is_frontend_specific(self):
        """The continuous front-end scores higher overall (0.809 vs 0.723 mean on the
        same positives), so it needs its own gate; the binary-era 0.40 left the
        provenance-relaxed gate low enough to admit 8 false fires where 0.50 admits 1."""
        from remove_ai_watermarks import doubao_engine

        assert doubao_engine._CONFIG.detect_ncc_threshold == 0.50

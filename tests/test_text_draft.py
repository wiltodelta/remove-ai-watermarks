"""text_draft: proposal-only OCR for verified-text manifests (no paddle needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from remove_ai_watermarks.text_draft import (
    TextDraft,
    choose_language,
    draft_available,
    draft_text_lines,
    group_word_boxes,
    stable_recognition,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestPureHelpers:
    def test_choose_language_follows_unicode_script(self):
        assert choose_language({"ch": ("你好", 0.9), "ru": ("xxx", 0.1), "en": ("yyy", 0.1)}) == "ch"
        assert choose_language({"ch": ("??", 0.1), "ru": ("привет", 0.9), "en": ("yyy", 0.1)}) == "ru"
        assert choose_language({"ch": ("??", 0.1), "ru": ("xxx", 0.1), "en": ("hello", 0.9)}) == "en"

    def test_stable_recognition_requires_identical_normalizations(self):
        # Same text modulo spacing/case (comma kept): stable.
        assert (
            stable_recognition([("Hello, World", 0.97), ("hello,world", 0.96), ("HELLO,  WORLD", 0.95)])
            == "Hello, World"
        )
        # Different texts: rejected (None).
        assert stable_recognition([("Hello", 0.97), ("Hallo", 0.96), ("Hello", 0.95)]) is None
        # Identical text but one read under the floor: rejected.
        assert stable_recognition([("Hello", 0.97), ("Hello", 0.60), ("Hello", 0.95)]) is None
        assert stable_recognition([("Hello", 0.97), ("Hello", 0.96), ("Hello", 0.95)], min_score=0.99) is None

    def test_group_word_boxes_merges_a_line_and_keeps_rows_apart(self):
        # Two words on one baseline merge; a distant lower line stays separate.
        merged = group_word_boxes([(10, 100, 60, 140), (70, 100, 120, 140)])
        assert len(merged) == 1
        assert merged[0] == (10, 100, 120, 140)
        apart = group_word_boxes([(10, 100, 60, 140), (10, 300, 60, 340)])
        assert len(apart) == 2


class TestDraftAvailable:
    def test_returns_a_bool_without_importing_paddle(self):
        assert isinstance(draft_available(), bool)


class _FakeDetector:
    """Emits Paddle-style pages for one synthetic poster."""

    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self._boxes = boxes

    def predict(self, _image: Any) -> Any:
        import numpy as np

        yield {
            "rec_boxes": np.asarray(self._boxes, dtype=np.float32),
            "rec_scores": [0.99] * len(self._boxes),
        }


class _FakeEngine:
    """Returns a fixed text at high score regardless of the crop."""

    def __init__(self, text: str) -> None:
        self._text = text

    def predict(self, _crop: Any) -> Any:
        yield {"rec_text": self._text, "rec_score": 0.97}


class _JitterEngine:
    """Changes its answer with the supplied crop dimensions - exactly what the gate rejects."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def predict(self, crop: Any) -> Any:
        text = self._texts[(crop.shape[1] // 8) % len(self._texts)]
        yield {"rec_text": text, "rec_score": 0.97}


class TestDraftTextLines:
    @staticmethod
    def _poster(tmp_path: Path) -> Path:
        path = tmp_path / "poster.png"
        Image.new("RGB", (400, 300), (255, 255, 255)).save(path)
        return path

    def test_accepts_crop_stable_and_rejects_jittered(self, tmp_path: Path):
        path = self._poster(tmp_path)
        stable = _FakeEngine("Invoice 42")
        jitter = _JitterEngine(["Hello", "Hallo", "Hullo"])
        # All three probe engines read the same string, so language = en; the
        # jitter engine then serves as the en engine and flips across crops.
        draft = draft_text_lines(
            path,
            detector=_FakeDetector([(20, 20, 200, 60), (20, 80, 200, 120)]),
            engines={"en": jitter, "ru": stable, "ch": stable},
        )
        assert isinstance(draft, TextDraft)
        assert len(draft.rejected) == 2
        assert all(line.language == "en" for line in draft.rejected)
        assert draft.accepted == ()

    def test_accepted_line_carries_box_text_script_and_floor(self, tmp_path: Path):
        path = self._poster(tmp_path)
        engine = _FakeEngine("Total: 1,234.56")
        draft = draft_text_lines(
            path,
            detector=_FakeDetector([(20, 20, 260, 60)]),
            engines={"en": engine, "ru": engine, "ch": engine},
        )
        (line,) = draft.accepted
        assert line.box == (20, 20, 260, 60)
        assert line.text == "Total: 1,234.56"
        assert line.script == "alphabetic"
        assert line.language == "en"
        assert line.min_score >= 0.85

    def test_cjk_probe_switches_script_and_language(self, tmp_path: Path):
        path = self._poster(tmp_path)
        cjk = _FakeEngine("每天都是一个新的机会。")
        latin = _FakeEngine("hello")
        draft = draft_text_lines(
            path,
            detector=_FakeDetector([(30, 30, 300, 90)]),
            engines={"en": latin, "ru": latin, "ch": cjk},
        )
        (line,) = draft.accepted
        assert line.language == "ch"
        assert line.script == "cjk"

    def test_unstable_geometry_draft_accepts_a_single_high_score_read(self, tmp_path: Path):
        path = self._poster(tmp_path)
        jitter = _JitterEngine(["Hello", "Hallo", "Hullo"])
        stable = _FakeEngine("Hello")
        # Probes disagree across engines so language is en; jitter would fail
        # the stable gate. Geometry mode uses the one en probe and accepts.
        draft = draft_text_lines(
            path,
            min_score=0.75,
            stable=False,
            detector=_FakeDetector([(20, 20, 200, 60)]),
            engines={"en": jitter, "ru": stable, "ch": stable},
        )
        (line,) = draft.accepted
        assert line.text == "Hallo"
        assert line.min_score >= 0.75


class TestActualCropJitter:
    @pytest.mark.parametrize(("language", "prefix"), [("en", "Latin"), ("ru", "Кириллица"), ("ch", "中文")])
    def test_changes_real_crops_and_rejects_content_dependent_reads(self, tmp_path, language, prefix):
        import numpy as np

        path = tmp_path / "gradient.png"
        image = np.broadcast_to(np.arange(240, dtype=np.uint8)[:, None, None], (240, 400, 3)).copy()
        Image.fromarray(image).save(path)

        class CropReader:
            def __init__(self):
                self.crops = []

            def predict(self, crop):
                self.crops.append(crop.copy())
                # A deterministic function of the supplied crop, never invocation order.
                yield {"rec_text": f"{prefix}{int(crop[0, 0, 0])}", "rec_score": 0.99}

        reader = CropReader()
        engines = {key: _FakeEngine("hello") for key in ("en", "ru", "ch")}
        engines[language] = reader
        draft = draft_text_lines(
            path,
            detector=_FakeDetector([(80, 80, 300, 180)]),
            engines=engines,
        )
        assert len(reader.crops) == 4  # Language probe plus three stability crops.
        assert len({crop.tobytes() for crop in reader.crops[1:]}) == 3
        assert draft.accepted == ()
        assert len(draft.rejected) == 1
        assert draft.rejected[0].language == language

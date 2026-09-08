"""The visible-mark example gallery is complete and self-consistent.

Two failures this suite exists to catch:
  * a mark registered without a committed example (the gallery lags the registry);
  * an engine that no longer detects its own canonical example (the gallery is
    generated from the engines' measured geometry, so this is a regression tripwire).

The examples are SYNTHETIC (``scripts/render_visible_examples.py`` composites the
committed silhouettes onto a generated base). Documented provider originals and
provider frames are separate regression inputs and remain byte-for-byte copies of
their sources.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from remove_ai_watermarks import watermark_registry as wr
from remove_ai_watermarks.image_io import imread
from remove_ai_watermarks.video import VIDEO_VISIBLE_MARKS, identify_video
from remove_ai_watermarks.video_visible import detect_sora_frame

_ROOT = Path(__file__).resolve().parents[1]
_GALLERY = _ROOT / "data" / "fixtures" / "visible"

_IMAGE_KEYS = [m.key for m in wr.known_marks()]
_PROVIDER_ORIGINALS = {
    "baidu": "provider-original.jpeg",
    "jimeng_pill": "provider-published-example.jpg",
    "kling": "provider-original-direct.png",
    "microsoft": "provider-original.png",
    "qwen": "provider-original.png",
    "yuanbao": "provider-original.png",
}
_VIDEO_PROVIDER_ORIGINALS = {
    "hailuo": "provider-original.mp4",
    "kling": "provider-original.mp4",
    "sora": "provider-original.mp4",
    "veo": "provider-original.mp4",
}


class TestGallery:
    def test_every_registered_mark_has_an_example(self) -> None:
        missing = [key for key in _IMAGE_KEYS if not (_GALLERY / key / "example.png").is_file()]
        assert missing == [], f"registered without an example: {missing}; run scripts/render_visible_examples.py"

    @pytest.mark.parametrize("key", _IMAGE_KEYS)
    def test_engine_detects_its_own_example(self, key: str) -> None:
        img = imread(str(_GALLERY / key / "example.png"))
        assert img is not None, key
        det = wr.get_mark(key).detect(img, provenance=key == "liblib_pill")
        assert det.detected, f"{key}: confidence {det.confidence:.3f} on its own example"

    def test_gallery_has_no_stray_directories(self) -> None:
        known = set(_IMAGE_KEYS) | set(VIDEO_VISIBLE_MARKS) | {"README.md"}
        extra = sorted(p.name for p in _GALLERY.iterdir() if p.name not in known)
        assert extra == [], f"gallery holds unregistered examples: {extra}; remove or register them"

    @pytest.mark.parametrize("key", sorted(_PROVIDER_ORIGINALS))
    def test_engine_detects_provider_original(self, key: str) -> None:
        img = imread(str(_GALLERY / key / _PROVIDER_ORIGINALS[key]))
        assert img is not None, key
        det = wr.get_mark(key).detect(img, provenance=False)
        assert det.detected, f"{key}: confidence {det.confidence:.3f} on provider original"


class TestVideoGallery:
    def test_every_registered_video_mark_has_an_example(self) -> None:
        missing = [key for key in VIDEO_VISIBLE_MARKS if not (_GALLERY / key / "example.mp4").is_file()]
        assert missing == [], f"video mark without an example: {missing}; run scripts/render_visible_examples.py"

    def test_selection_accepts_each_example(self) -> None:
        # The shipped temporal selection (not just the per-frame detector) must
        # accept the clip: table order resolves cross-template ties, so the example
        # must carry the discriminative variant of its mark.
        for key in VIDEO_VISIBLE_MARKS:
            rep = identify_video(_GALLERY / key / "example.mp4", check_visible=True)
            assert rep.visible_mark == key, f"{key}: selection returned {rep.visible_mark!r}"

    @pytest.mark.parametrize("key", sorted(_VIDEO_PROVIDER_ORIGINALS))
    def test_selection_accepts_provider_original(self, key: str) -> None:
        rep = identify_video(_GALLERY / key / _VIDEO_PROVIDER_ORIGINALS[key], check_visible=True)
        assert rep.visible_mark == key, f"{key}: selection returned {rep.visible_mark!r}"

    def test_sora_frame_localizer_accepts_provider_frame(self) -> None:
        img = imread(str(_GALLERY / "sora" / "provider-frame.jpg"))
        assert img is not None
        det = detect_sora_frame(img)
        assert det.confidence >= 0.60, f"sora: confidence {det.confidence:.3f} on provider frame"

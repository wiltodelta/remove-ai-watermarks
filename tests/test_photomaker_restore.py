"""Tests for the PhotoMaker-V2 face identity restoration helper.

These tests cover the pure-Python parts (face crop math, composite, the no-faces
no-op, the is_available guard) WITHOUT loading PhotoMaker or SDXL -- the model-loading
path is gated behind ``is_available()`` and exercised manually via the Modal cert
sweep, mirroring the convention used for ``face_restore`` and ``upscaler``.

The end-to-end PhotoMaker run is monkey-patched: we replace ``_get_pipeline`` with a
fake pipeline whose ``__call__`` returns a known constant-color face, so we can verify
that the right boxes get the right pixels composited back.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from remove_ai_watermarks import photomaker_restore


class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(photomaker_restore.is_available(), bool)


class TestFaceCropSquare:
    def test_centers_on_face_box(self):
        img = np.full((400, 400, 3), 128, dtype=np.uint8)
        crop, box = photomaker_restore._face_crop_square(img, (100, 150, 80, 80))
        x1, y1, x2, y2 = box
        # The crop covers the requested box (with padding)
        assert x1 <= 100
        assert y1 <= 150
        assert x2 >= 180
        assert y2 >= 230
        assert crop.shape[0] == y2 - y1
        assert crop.shape[1] == x2 - x1

    def test_clips_at_image_edges(self):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        crop, (x1, y1, x2, y2) = photomaker_restore._face_crop_square(img, (180, 180, 30, 30))
        # Box must be clipped within the image
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 200
        assert y2 <= 200
        assert crop.shape[0] == y2 - y1
        assert crop.shape[1] == x2 - x1

    def test_pad_widens_the_crop(self):
        img = np.full((400, 400, 3), 128, dtype=np.uint8)
        _, no_pad = photomaker_restore._face_crop_square(img, (150, 150, 50, 50), pad=0.0)
        _, with_pad = photomaker_restore._face_crop_square(img, (150, 150, 50, 50), pad=0.5)
        assert (with_pad[2] - with_pad[0]) > (no_pad[2] - no_pad[0])


class TestCompositeFaces:
    def test_empty_list_returns_base_unchanged(self):
        base = np.full((100, 100, 3), 64, dtype=np.uint8)
        out = photomaker_restore._composite_faces(base, [])
        assert np.array_equal(out, base)

    def test_box_outside_image_is_skipped(self):
        base = np.full((100, 100, 3), 64, dtype=np.uint8)
        crop = np.full((40, 40, 3), 200, dtype=np.uint8)
        out = photomaker_restore._composite_faces(base, [(crop, (200, 200, 240, 240))])
        assert np.array_equal(out, base)

    def test_composited_box_pulls_pixel_value_toward_crop(self):
        base = np.full((200, 200, 3), 40, dtype=np.uint8)
        crop = np.full((50, 50, 3), 220, dtype=np.uint8)
        # Place the crop fully inside the image at (60, 60)..(110, 110)
        out = photomaker_restore._composite_faces(base, [(crop, (60, 60, 110, 110))])
        # The box center should be heavily biased toward the crop color (>120) ...
        assert out[85, 85, 0] > 120
        # ... and corners (well outside the feathered region) stay close to base
        assert int(out[0, 0, 0]) - int(base[0, 0, 0]) <= 1


class TestRestoreFacesPhotomakerControlFlow:
    """End-to-end control flow with a fake pipeline -- no diffusion model loaded."""

    @staticmethod
    def _fake_pipeline_class(fill_value: int = 200):
        """Class-based fake (no ``__call__`` on a SimpleNamespace, which Python won't dispatch)."""
        from PIL import Image

        size = photomaker_restore._PHOTOMAKER_FACE_SIZE
        fake_face = Image.fromarray(np.full((size, size, 3), fill_value, dtype=np.uint8))

        class _FakePipe:
            device = "cpu"

            def __call__(self, **_kwargs):
                return SimpleNamespace(images=[fake_face])

        return _FakePipe()

    def test_no_faces_returns_cleaned_unchanged(self, monkeypatch):
        # Force is_available so we never hit the missing-extra branch
        monkeypatch.setattr(photomaker_restore, "is_available", lambda: True)
        monkeypatch.setattr(photomaker_restore, "_get_pipeline", lambda: self._fake_pipeline_class())

        orig = np.full((200, 200, 3), 30, dtype=np.uint8)
        cleaned = np.full((200, 200, 3), 90, dtype=np.uint8)
        out = photomaker_restore.restore_faces_photomaker(orig, cleaned, detect_faces_fn=lambda _b: [])
        assert np.array_equal(out, cleaned)

    def test_one_face_gets_composited_into_cleaned(self, monkeypatch):
        monkeypatch.setattr(photomaker_restore, "is_available", lambda: True)
        monkeypatch.setattr(photomaker_restore, "_get_pipeline", lambda: self._fake_pipeline_class(fill_value=210))

        orig = np.full((400, 400, 3), 30, dtype=np.uint8)
        cleaned = np.full((400, 400, 3), 90, dtype=np.uint8)
        # Mark the original face region with a distinctive color so we can confirm the
        # crop reached the pipeline (not strictly tested here, but useful sanity).
        cv2.rectangle(orig, (150, 150), (250, 250), (200, 100, 50), -1)

        out = photomaker_restore.restore_faces_photomaker(
            orig, cleaned, detect_faces_fn=lambda _b: [(150, 150, 100, 100)]
        )
        # The cleaned image should have shifted toward the fake-face fill (210) inside
        # the face region.
        assert out[200, 200, 0] > 150
        # And the corner pixels (well outside the feather) should still be near the base.
        assert int(out[0, 0, 0]) - int(cleaned[0, 0, 0]) <= 1

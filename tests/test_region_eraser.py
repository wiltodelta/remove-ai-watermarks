"""Tests for the universal region eraser."""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks.region_eraser import boxes_to_mask, erase, lama_available, migan_available


class TestPaddedCropBox:
    """The padded bounding box that bounds the learned backends' ONNX working set."""

    def test_empty_mask_returns_none(self):
        from remove_ai_watermarks.region_eraser import _padded_crop_box

        assert _padded_crop_box(np.zeros((100, 100), np.uint8), 100, 100, pad_frac=0.1, pad_min=8) is None

    def test_pad_min_dominates_and_clamps_at_border(self):
        from remove_ai_watermarks.region_eraser import _padded_crop_box

        mask = np.zeros((100, 100), np.uint8)
        mask[0:5, 0:5] = 255  # 5-px mark in the top-left corner
        # pad = max(8, int(0.1*5)) = 8; x0 clamps to 0 (not -8), x1 = min(100, 4+1+8) = 13.
        assert _padded_crop_box(mask, 100, 100, pad_frac=0.1, pad_min=8) == (0, 0, 13, 13)

    def test_pad_frac_dominates_for_large_mark(self):
        from remove_ai_watermarks.region_eraser import _padded_crop_box

        mask = np.zeros((400, 400), np.uint8)
        mask[100:300, 100:300] = 255  # 200-px span
        # pad = max(8, int(0.2*200)) = 40; box = (100-40, .., 299+1+40, ..).
        assert _padded_crop_box(mask, 400, 400, pad_frac=0.2, pad_min=8) == (60, 60, 340, 340)

    def test_clamps_at_far_border(self):
        from remove_ai_watermarks.region_eraser import _padded_crop_box

        mask = np.zeros((50, 60), np.uint8)
        mask[45:50, 55:60] = 255  # bottom-right corner
        _x0, _y0, x1, y1 = _padded_crop_box(mask, 50, 60, pad_frac=0.1, pad_min=8)
        assert x1 == 60  # clamped to w, no overflow
        assert y1 == 50  # clamped to h, no overflow


class TestBoxesToMask:
    def test_mask_set_inside_box(self):
        mask = boxes_to_mask((100, 100), [(10, 20, 30, 40)], dilate=0)
        assert mask[25, 15] == 255  # inside
        assert mask[0, 0] == 0  # outside
        assert mask.shape == (100, 100)

    def test_multiple_boxes(self):
        mask = boxes_to_mask((100, 100), [(0, 0, 10, 10), (90, 90, 10, 10)], dilate=0)
        assert mask[5, 5] == 255
        assert mask[95, 95] == 255
        assert mask[50, 50] == 0

    def test_dilate_grows_mask(self):
        m0 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=0)
        m5 = boxes_to_mask((100, 100), [(40, 40, 10, 10)], dilate=5)
        assert m5.sum() > m0.sum()

    def test_box_clipped_to_bounds(self):
        # box partly outside the image must not raise and stays in-bounds
        mask = boxes_to_mask((50, 50), [(40, 40, 100, 100)], dilate=0)
        assert mask[45, 45] == 255


class TestEraseCv2:
    def _image_with_logo(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        img = np.full((200, 200, 3), 120, np.uint8)  # flat gray background
        box = (140, 160, 50, 30)
        x, y, w, h = box
        img[y : y + h, x : x + w] = (255, 255, 255)  # bright "logo"
        return img, box

    def test_erase_changes_region(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2")
        x, y, w, h = box
        # on a flat background the logo region should be repainted near gray
        region = out[y : y + h, x : x + w]
        assert abs(float(region.mean()) - 120) < 20
        assert not np.array_equal(out, img)

    def test_pixels_outside_box_untouched(self):
        img, box = self._image_with_logo()
        out = erase(img, boxes=[box], backend="cv2", dilate=0)
        # a far corner must be identical
        assert np.array_equal(img[:50, :50], out[:50, :50])

    def test_no_boxes_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, boxes=[], backend="cv2")
        assert np.array_equal(img, out)

    def test_empty_mask_returns_copy(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        out = erase(img, mask=np.zeros((100, 100), np.uint8), backend="cv2")
        assert np.array_equal(img, out)


class TestNonBgrInputs:
    """cv2.inpaint rejects 4-channel BGRA and 2D-only entry points must work."""

    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="cv2")
        assert out.shape == gray.shape

    def test_bgra_preserves_alpha_and_does_not_raise(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200  # opaque-ish alpha plane
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="cv2", dilate=0)
        assert out.shape == bgra.shape
        # alpha plane is carried through unchanged
        assert np.array_equal(out[..., 3], bgra[..., 3])


class TestBackendTable:
    """The fill-backend names are stated in several places; they must agree.

    They are deliberately separate LITERALS rather than one derived list: deriving the
    CLI choices or the registry's ``Backend`` from ``region_eraser`` would give
    ``watermark_registry`` (and therefore every ``--help`` and every metadata-only
    ``identify``) a module-level cv2 import. This test keeps the copies in sync instead.
    """

    def test_registry_literal_matches_the_eraser_table(self):
        import typing

        from remove_ai_watermarks import region_eraser, watermark_registry

        assert set(typing.get_args(watermark_registry.Backend)) == set(region_eraser.FILL_BACKENDS)

    def test_executable_backends_are_the_table_minus_auto(self):
        import typing

        from remove_ai_watermarks import region_eraser

        assert set(typing.get_args(region_eraser.Backend)) == set(region_eraser.FILL_BACKENDS) - {"auto"}

    def test_learned_backends_name_real_module_attributes(self):
        from remove_ai_watermarks import region_eraser

        for name, row in region_eraser._LEARNED_BACKENDS.items():
            assert name in region_eraser.FILL_BACKENDS
            assert callable(getattr(region_eraser, row.available))
            assert callable(getattr(region_eraser, row.erase))

    def test_unknown_backend_degrades_to_cv2_instead_of_raising(self):
        """``erase`` is public: a library caller passing ``"auto"`` (or a typo) must get
        the classical fill, not a KeyError."""
        img = np.full((64, 64, 3), 100, np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:40, 20:40] = 255
        assert erase(img, mask=mask, backend="auto").shape == img.shape  # type: ignore[arg-type]


class TestLamaBackend:
    def test_lama_raises_when_unavailable(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        if lama_available():
            pytest.skip("onnxruntime installed; cannot test the unavailable path")
        with pytest.raises(RuntimeError, match="onnxruntime"):
            erase(img, boxes=[(10, 10, 20, 20)], backend="lama")


class TestLamaChannelHandling:
    """erase_lama must accept grayscale (2D) and BGRA (4-channel) like erase_cv2.

    The real ONNX model is never loaded -- the session is faked to an identity
    inpaint, so this exercises only the channel promote/split wrapper (the fix for
    LaMa crashing on grayscale and dropping alpha on BGRA).
    """

    @pytest.fixture
    def _fake_lama(self, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import region_eraser

        class _In:
            def __init__(self, name: str, shape: list[int]):
                self.name = name
                self.shape = shape

        class _FakeSession:
            def get_inputs(self):
                return [_In("image", [1, 3, 512, 512]), _In("mask", [1, 1, 512, 512])]

            def run(self, _outputs, feeds):
                # Identity inpaint: echo the image tensor (1,3,size,size) back.
                return [feeds["image"]]

        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        monkeypatch.setattr(region_eraser, "_get_lama_session", lambda: _FakeSession())

    @pytest.mark.usefixtures("_fake_lama")
    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="lama")
        assert out.ndim == 2
        assert out.shape == gray.shape

    @pytest.mark.usefixtures("_fake_lama")
    def test_bgra_preserves_alpha(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200  # opaque-ish alpha plane
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="lama")
        assert out.shape == bgra.shape
        assert np.array_equal(out[..., 3], bgra[..., 3])  # alpha carried through unchanged


class TestMiganBackend:
    def test_migan_raises_when_unavailable(self):
        img = np.full((100, 100, 3), 50, np.uint8)
        if migan_available():
            pytest.skip("onnxruntime installed; cannot test the unavailable path")
        with pytest.raises(RuntimeError, match="onnxruntime"):
            erase(img, boxes=[(10, 10, 20, 20)], backend="migan")


class TestMiganWrapper:
    """erase_migan without the real model: fake session returns a solid-red field
    and captures the fed mask. Exercises the mask-polarity inversion, masked-only
    compositing, and grayscale/BGRA channel handling."""

    captured: dict

    @pytest.fixture
    def _fake_migan(self, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import region_eraser

        self.captured = {}

        class _In:
            def __init__(self, name: str):
                self.name = name

        class _FakeSession:
            def __init__(self, outer):
                self.outer = outer

            def get_inputs(self):
                return [_In("image"), _In("mask")]

            def run(self, _outputs, feeds):
                self.outer.captured["mask"] = feeds["mask"]
                self.outer.captured["image_shape"] = feeds["image"].shape
                img = feeds["image"]  # (1,3,H,W) RGB
                red = np.zeros_like(img)
                red[:, 0] = 255  # pure red in RGB
                return [red]

        monkeypatch.setattr(region_eraser, "migan_available", lambda: True)
        monkeypatch.setattr(region_eraser, "_get_migan_session", lambda: _FakeSession(self))

    @pytest.mark.usefixtures("_fake_migan")
    def test_composites_only_masked_region_and_inverts_mask(self):
        img = np.full((100, 100, 3), 120, np.uint8)  # BGR
        out = erase(img, boxes=[(40, 40, 20, 20)], backend="migan", dilate=0)
        # inside the box -> red (BGR (0,0,255)); outside -> untouched
        assert tuple(int(v) for v in out[50, 50]) == (0, 0, 255)
        assert np.array_equal(out[:30, :30], img[:30, :30])
        # mask fed to MI-GAN is inverted: 0 (hole) inside the box, 255 (known) outside
        m = self.captured["mask"][0, 0]
        assert m[50, 50] == 0
        assert m[10, 10] == 255

    @pytest.mark.usefixtures("_fake_migan")
    def test_crops_around_mask_so_onnx_input_is_bounded(self):
        # Large frame, small corner mark: the tensor fed to MI-GAN is the padded
        # CROP (pad = max(256, 2*bbox)), not the full image -- this is what holds the
        # ONNX working set roughly constant on big uploads instead of scaling with
        # the image (the memory fix). Untouched pixels stay exact; the mark is filled.
        img = np.full((2000, 3000, 3), 120, np.uint8)
        out = erase(img, boxes=[(2900, 1900, 60, 60)], backend="migan", dilate=0)
        assert out.shape == img.shape
        _, _, fh, fw = self.captured["image_shape"]
        assert fh < 700  # crop height, not the 2000px frame
        assert fw < 700  # crop width, not the 3000px frame
        assert tuple(int(v) for v in out[1930, 2930]) == (0, 0, 255)  # mark -> red fill
        assert np.array_equal(out[:100, :100], img[:100, :100])  # far corner untouched

    @pytest.mark.usefixtures("_fake_migan")
    def test_grayscale_2d_does_not_raise(self):
        gray = np.full((100, 100), 120, np.uint8)
        out = erase(gray, boxes=[(40, 40, 20, 20)], backend="migan", dilate=0)
        assert out.ndim == 2
        assert out.shape == gray.shape

    @pytest.mark.usefixtures("_fake_migan")
    def test_bgra_preserves_alpha(self):
        bgra = np.full((100, 100, 4), 120, np.uint8)
        bgra[..., 3] = 200
        out = erase(bgra, boxes=[(40, 40, 20, 20)], backend="migan", dilate=0)
        assert out.shape == bgra.shape
        assert np.array_equal(out[..., 3], bgra[..., 3])


class TestSixteenBitInputs:
    """A 16-bit source must survive the fill, whatever the backend can hold.

    cv2.inpaint takes 16-bit only as a single channel (its colour path is 8-bit)
    and the MI-GAN ONNX declares a uint8 input tensor, so those two fill through an
    8-bit copy; LaMa is float32 and carries the depth end to end. In every case the
    pixels OUTSIDE the mask stay bit-exact at the source's depth -- feeding a 16-bit
    colour image to cv2 used to raise a bare OpenCV "Unsupported format" error.
    """

    @staticmethod
    def _gradient16() -> np.ndarray:
        column = np.linspace(0, 65535, 96, dtype=np.uint16)
        return np.repeat(np.tile(column, (96, 1))[:, :, None], 3, axis=2).copy()

    def test_cv2_fills_a_16_bit_image_without_raising(self):
        img = self._gradient16()
        out = erase(img, boxes=[(20, 20, 30, 30)], backend="cv2", dilate=3)
        assert out.dtype == np.uint16
        assert out.shape == img.shape

    def test_cv2_leaves_everything_outside_the_mask_bit_exact(self):
        img = self._gradient16()
        mask = boxes_to_mask(img.shape[:2], [(20, 20, 30, 30)], dilate=3)
        out = erase(img, mask=mask, backend="cv2")
        keep = mask <= 127
        assert np.array_equal(out[keep], img[keep])
        assert not np.array_equal(out[mask > 127], img[mask > 127])

    def test_the_fill_lands_in_the_source_s_range_not_the_8_bit_one(self):
        # The narrow-then-widen round trip must scale back up: leaving the 8-bit
        # levels in a uint16 array would paint the mask near-black.
        img = np.full((96, 96, 3), 40000, np.uint16)
        mask = boxes_to_mask(img.shape[:2], [(30, 30, 20, 20)], dilate=3)
        filled = erase(img, mask=mask, backend="cv2")[mask > 127]
        assert filled.min() > 30000

    def test_a_float_image_says_what_is_wrong(self):
        img = np.full((64, 64, 3), 0.5, np.float32)
        with pytest.raises(RuntimeError, match="integer images"):
            erase(img, boxes=[(10, 10, 20, 20)], backend="cv2")


class TestLamaSixteenBit:
    """LaMa normalises by the SOURCE's full scale, so 16 bits go through intact."""

    @pytest.fixture
    def fake_lama(self, monkeypatch: pytest.MonkeyPatch):
        from remove_ai_watermarks import region_eraser

        class _In:
            def __init__(self, name: str, shape: list[int]):
                self.name = name
                self.shape = shape

        seen: dict[str, float] = {}

        class _FakeSession:
            def get_inputs(self):
                return [_In("image", [1, 3, 512, 512]), _In("mask", [1, 1, 512, 512])]

            def run(self, _outputs, feeds):
                seen["max"] = float(feeds["image"].max())
                return [feeds["image"]]

        monkeypatch.setattr(region_eraser, "lama_available", lambda: True)
        monkeypatch.setattr(region_eraser, "_get_lama_session", lambda: _FakeSession())
        return seen

    def test_the_model_sees_a_normalised_tensor_and_the_output_keeps_its_depth(self, fake_lama):
        img = np.full((96, 96, 3), 60000, np.uint16)
        out = erase(img, boxes=[(30, 30, 20, 20)], backend="lama")
        # Dividing a 16-bit crop by 255 fed the model ~235 instead of ~0.92, and the
        # uint8 cast on the way out then wrapped it into near-black pixels.
        assert fake_lama["max"] <= 1.0
        assert out.dtype == np.uint16
        assert out[45, 45].max() > 30000

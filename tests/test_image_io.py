"""Tests for Unicode-safe cv2 image IO (issue #17)."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from pathlib import Path

# Non-ASCII filenames that break cv2.imread/imwrite on Windows (issue #17).
_UNICODE_NAMES = [
    "jimeng-2026-05-27-一面白色的墙.png",  # Chinese
    "тест-изображение.png",  # Cyrillic
    "café-señor.png",  # accented Latin
]


def _make_bgr() -> np.ndarray:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = (10, 120, 240)  # a BGR block so the round-trip is checkable
    return img


class TestUnicodeRoundTrip:
    def test_write_then_read_preserves_pixels(self, tmp_path: Path) -> None:
        for name in _UNICODE_NAMES:
            path = tmp_path / name
            src = _make_bgr()
            assert image_io.imwrite(path, src) is True
            assert path.exists()
            out = image_io.imread(path)
            assert out is not None
            # PNG is lossless: pixels must match exactly.
            assert np.array_equal(out, src)

    def test_alpha_round_trip_with_unchanged_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "豆包-alpha.png"
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[..., 3] = 128
        assert image_io.imwrite(path, bgra) is True
        out = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert out is not None
        assert out.shape[2] == 4
        assert np.array_equal(out, bgra)

    def test_reads_file_written_by_raw_cv2(self, tmp_path: Path) -> None:
        # An ASCII file written by plain cv2 must read back identically through
        # the wrapper (decode path is byte-compatible with cv2.imread).
        path = tmp_path / "ascii.png"
        src = _make_bgr()
        cv2.imwrite(str(path), src)
        out = image_io.imread(path)
        assert out is not None
        assert np.array_equal(out, src)


class TestToBgr:
    def test_grayscale_2d_promoted_to_bgr(self) -> None:
        gray = np.full((4, 5), 120, dtype=np.uint8)
        out = image_io.to_bgr(gray)
        assert out.shape == (4, 5, 3)
        # GRAY2BGR replicates the channel, so all three match the source.
        assert np.array_equal(out[..., 0], gray)
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_single_channel_3d_promoted(self) -> None:
        gray = np.full((4, 5, 1), 7, dtype=np.uint8)
        assert image_io.to_bgr(gray).shape == (4, 5, 3)

    def test_bgra_dropped_to_bgr(self) -> None:
        bgra = np.zeros((4, 5, 4), dtype=np.uint8)
        bgra[..., :3] = (10, 120, 240)
        out = image_io.to_bgr(bgra)
        assert out.shape == (4, 5, 3)
        assert np.array_equal(out, bgra[..., :3])

    def test_bgr_returned_unchanged(self) -> None:
        bgr = _make_bgr()
        out = image_io.to_bgr(bgr)
        assert out is bgr  # 3-channel: no copy


class TestFailureSemantics:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert image_io.imread(tmp_path / "does-not-exist-不存在.png") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        assert image_io.imread(path) is None

    def test_undecodable_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.png"
        path.write_bytes(b"not an image")
        assert image_io.imread(path) is None

    def test_imwrite_to_missing_directory_returns_false(self, tmp_path: Path) -> None:
        # An unwritable path must return False (cv2.imwrite contract), not raise.
        path = tmp_path / "no-such-dir" / "out.png"
        assert image_io.imwrite(path, _make_bgr()) is False


def _avif_writable() -> bool:
    """True when the installed Pillow can ENCODE AVIF (needed to synthesize a fixture);
    read support is what we test, but we need a writer to make the sample."""
    from PIL import Image, features

    if not (hasattr(features, "check") and features.check("avif")):
        return False
    try:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(io.BytesIO(), format="AVIF")
        return True
    except Exception:
        return False


_HAS_AVIF_WRITE = _avif_writable()


class TestPillowFallback:
    """OpenCV cannot decode HEIC/AVIF; :func:`imread` falls back to Pillow (with the
    pillow-heif plugin) so the pixel/removal path reads them. Uses a synthetic AVIF
    (no corpus files); HEIC is exercised the same way in production."""

    @pytest.mark.skipif(not _HAS_AVIF_WRITE, reason="Pillow AVIF encoder unavailable")
    def test_avif_decodes_via_fallback(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "x.avif"
        Image.fromarray(np.full((32, 48, 3), (10, 20, 200), np.uint8), "RGB").save(path)
        img = image_io.imread(path)  # cv2 fails on AVIF -> Pillow fallback
        assert img is not None
        assert img.shape == (32, 48, 3)
        # Source RGB (R=10, G=20, B=200) -> BGR, so blue [...,0] is high, red [...,2] low
        # (thresholds loose for AVIF's lossy compression).
        assert int(img[0, 0, 0]) > 150
        assert int(img[0, 0, 2]) < 70

    @pytest.mark.skipif(not _HAS_AVIF_WRITE, reason="Pillow AVIF encoder unavailable")
    def test_avif_alpha_survives_read_bgr_and_alpha(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "x.avif"
        rgba = np.dstack([np.full((20, 20, 3), 80, np.uint8), np.full((20, 20), 128, np.uint8)])
        Image.fromarray(rgba, "RGBA").save(path)
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert bgr is not None
        assert bgr.shape == (20, 20, 3)
        assert alpha is not None  # the source alpha plane survives the fallback
        assert alpha.shape == (20, 20)


def _heif_writable(fmt: str) -> bool:
    try:
        image_io._register_heif()
        from PIL import Image

        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(io.BytesIO(), format=fmt, quality=100)
        return True
    except Exception:
        return False


class TestQualityPreservingWrite:
    """The removal only touches the mark footprint, so the container re-encode must
    not degrade the untouched pixels: JPEG/WebP are written at max quality, HEIC/AVIF
    (which cv2 cannot encode) via Pillow instead of crashing."""

    def test_jpeg_written_near_lossless(self, tmp_path: Path) -> None:
        # a smooth gradient -> JPEG at quality 100 / 4:4:4 is near-lossless
        g = np.tile(np.linspace(30, 210, 64, dtype=np.uint8), (48, 1))
        img = np.dstack([g, g, g])
        p = tmp_path / "x.jpg"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert float(np.abs(img.astype(int) - back.astype(int)).mean()) < 1.0

    def test_webp_written_lossless(self, tmp_path: Path) -> None:
        # Regression: cv2 WebP quality 1-100 is LOSSY; lossless needs > 100. A
        # mark-removal .webp re-encode must NOT degrade the untouched pixels, so
        # a full-frame round-trip of random data must be bit-identical.
        img = np.random.default_rng(0).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        p = tmp_path / "x.webp"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert np.array_equal(back, img), "WebP re-encode was lossy"

    @pytest.mark.skipif(not _heif_writable("HEIF"), reason="no HEIC encoder in this env")
    def test_heic_write_roundtrips(self, tmp_path: Path) -> None:
        # cv2 cannot encode HEIC (used to raise); imwrite must route through Pillow.
        img = np.full((32, 48, 3), (30, 140, 200), np.uint8)
        p = tmp_path / "x.heic"
        assert image_io.imwrite(p, img) is True
        back = image_io.imread(p)
        assert back is not None
        assert back.shape == (32, 48, 3)

    def test_unencodable_ext_returns_false_not_raises(self, tmp_path: Path) -> None:
        # a bogus extension cv2 can't encode returns False (never raises the cv2.error).
        assert image_io.imwrite(tmp_path / "x.zzz", _make_bgr()) is False


class TestDisplayTagCarry:
    """A re-encode must not eat the source's ICC profile and EXIF orientation
    (issue #98): cv2's encoders write no container metadata, so erase/visible/
    invisible outputs used to come back colourless and, on a build that leaves the
    raster stored-orientation, sideways. The pixel paths decode with
    IMREAD_UNCHANGED (which never rotates), so the raster keeps the source's stored
    orientation and the tag must ride along to the output."""

    ORIENT = 6  # Rotate 90 CW: a portrait display of a landscape-stored raster.

    @staticmethod
    def _icc() -> bytes:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    @classmethod
    def _source(cls, path: Path, *, orient: int | None = 6, icc: bool = True) -> Path:
        """A 96x64 stored raster carrying an sRGB profile and Orientation=6."""
        import piexif
        from PIL import Image

        kwargs: dict[str, object] = {"icc_profile": cls._icc()} if icc else {}
        if orient is not None:
            exif = piexif.dump({"0th": {piexif.ImageIFD.Orientation: orient, piexif.ImageIFD.Make: "TestCam"}})
            if path.suffix in (".png", ".webp"):
                kwargs["exif"] = exif  # Pillow writes the eXIf chunk / WebP EXIF chunk
        Image.new("RGB", (96, 64), "red").save(path, **kwargs)
        if orient is not None and path.suffix in (".jpg", ".jpeg"):
            piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: orient}}), str(path))
        return path

    @staticmethod
    def _tags(path: Path) -> tuple[object, object]:
        from PIL import Image

        with Image.open(path) as im:
            return im.info.get("icc_profile"), im.getexif().get(0x0112)

    def _stored_raster(self, path: Path) -> np.ndarray:
        # What the pixel paths hold after reading the source: UNCHANGED never rotates.
        raster = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert raster is not None
        return raster

    def test_png_output_keeps_profile_and_orientation(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT
        # The splice must not touch pixels: Pillow decodes without rotating.
        from PIL import Image

        with Image.open(out) as im:
            assert np.array_equal(np.asarray(im)[:, :, ::-1], raster)

    def test_jpeg_output_carries_both_tags(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.jpg"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT

    def test_webp_output_carries_both_tags_losslessly(self, tmp_path: Path) -> None:
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.webp"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert icc == source_icc
        assert orient == self.ORIENT
        # The lossless Pillow re-save must stay pixel-identical.
        back = self._stored_raster(out)
        assert back is not None
        assert np.array_equal(back, raster)

    def test_webp_source_orientation_travels(self, tmp_path: Path) -> None:
        # IMREAD_UNCHANGED preserves stored pixels, so even a fresh PNG output must be
        # tagged, or the portrait source would display sideways.
        src = self._source(tmp_path / "src.webp")
        raster = self._stored_raster(src)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        assert self._tags(out)[1] == self.ORIENT

    def test_an_upright_raster_is_not_tagged_into_a_second_rotation(self, tmp_path: Path) -> None:
        # A raster already turned upright must not receive the tag again. Stored 96x64
        # with Orientation 6 displays as 64x96, so the upright raster is 64 wide.
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        upright = np.zeros((96, 64, 3), np.uint8)  # 64 cols x 96 rows: stored dims swapped
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, upright, display_tags_from=src) is True
        icc, orient = self._tags(out)
        assert orient is None
        assert icc == source_icc

    def test_a_raster_of_unexplained_shape_is_not_tagged(self, tmp_path: Path) -> None:
        # A resized pipeline output matches neither the stored nor the upright
        # dimensions; carrying the tag there would be a guess, so it is dropped.
        src = self._source(tmp_path / "src.png")
        resized = np.zeros((31, 17, 3), np.uint8)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, resized, display_tags_from=src) is True
        assert self._tags(out)[1] is None

    def test_an_untagged_source_changes_nothing(self, tmp_path: Path) -> None:
        # No tags to carry: the output must be byte-identical to a plain write.
        src = self._source(tmp_path / "src.png", orient=None, icc=False)
        raster = self._stored_raster(src)
        tagged, plain = tmp_path / "tagged.png", tmp_path / "plain.png"
        assert image_io.imwrite(tagged, raster, display_tags_from=src) is True
        assert image_io.imwrite(plain, raster) is True
        assert tagged.read_bytes() == plain.read_bytes()

    def test_a_missing_decode_source_still_writes(self, tmp_path: Path) -> None:
        raster = np.zeros((8, 8, 3), np.uint8)
        out = tmp_path / "out.png"
        assert image_io.imwrite(out, raster, display_tags_from=tmp_path / "gone.png") is True
        assert out.exists()
        assert self._tags(out) == (None, None)

    def test_in_place_rewrite_keeps_the_tags(self, tmp_path: Path) -> None:
        # Tags are read BEFORE the write, so rewriting a file over itself cannot
        # lose what it is about to restore.
        path = self._source(tmp_path / "in.png")
        source_icc, _ = self._tags(path)
        raster = self._stored_raster(path)
        assert image_io.imwrite(path, raster, display_tags_from=path) is True
        icc, orient = self._tags(path)
        assert icc == source_icc
        assert orient == self.ORIENT

    def test_png_splice_replaces_rather_than_stacks(self, tmp_path: Path) -> None:
        # Rewriting an already-tagged PNG must not duplicate the chunks.
        from remove_ai_watermarks.image_io import _png_splice_display_tags

        base = tmp_path / "base.png"
        assert image_io.imwrite(base, np.zeros((8, 8, 3), np.uint8)) is True
        once = _png_splice_display_tags(base.read_bytes(), self._icc(), self.ORIENT)
        twice = _png_splice_display_tags(once, self._icc(), self.ORIENT)

        kinds: list[bytes] = []
        i = 8
        while i + 8 <= len(twice):
            length = int.from_bytes(twice[i : i + 4], "big")
            kinds.append(twice[i + 4 : i + 8])
            i += 12 + length
        assert kinds.count(b"iCCP") == 1
        assert kinds.count(b"eXIf") == 1

    @pytest.mark.skipif(not _heif_writable("HEIF"), reason="no HEIC encoder in this env")
    def test_heif_output_carries_the_profile_and_displays_upright(self, tmp_path: Path) -> None:
        # Pillow's HEIF writer bakes the orientation into the pixels on save (it
        # transposes the raster and writes Orientation 1), so the display result,
        # not the raw tag, is what to assert: upright portrait, exact profile.
        src = self._source(tmp_path / "src.png")
        source_icc, _ = self._tags(src)
        raster = self._stored_raster(src)
        out = tmp_path / "out.heic"
        assert image_io.imwrite(out, raster, display_tags_from=src) is True
        from PIL import Image

        with Image.open(out) as im:
            assert im.info.get("icc_profile") == source_icc
            assert im.size == (64, 96)  # the upright portrait, not the stored landscape
            assert im.getexif().get(0x0112) in (None, 1)


class TestRemoverWriteCarriesDisplayTags:
    """The diffusion path's write must thread ``display_tags_from`` through to
    ``imwrite``: the invisible pipeline decodes via Pillow (never rotated), saves a
    temp PNG that carries the ICC profile but no orientation (``exif_transpose``
    consumed it upstream), and ``_write_output`` restores from that input path."""

    def test_the_output_mirrors_its_input_path(self, tmp_path: Path) -> None:
        from PIL import Image, ImageCms

        from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        # Exactly what the invisible engine's temp save produces: ICC, no orientation.
        temp = tmp_path / "temp.png"
        Image.new("RGB", (32, 16), "blue").save(temp, icc_profile=icc)

        remover = WatermarkRemover.__new__(WatermarkRemover)  # _write_output uses no state
        out = tmp_path / "out.png"
        remover._write_output(Image.new("RGB", (32, 16), "blue"), out, display_tags_from=temp)

        with Image.open(out) as im:
            assert im.info.get("icc_profile") == icc
            assert im.getexif().get(0x0112) is None

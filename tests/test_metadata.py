"""Tests for AI metadata detection and removal."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import piexif
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.metadata import (
    C2PA_UUID,
    _is_ai_key,
    c2pa_marker_in,
    exif_generator,
    get_ai_metadata,
    has_ai_metadata,
    iptc_ai_system,
    remove_ai_metadata,
    samsung_genai,
    synthid_source,
    xai_signature,
)

# Real, committed C2PA sample images used to ground the SynthID-source tests.
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# ── Key detection ───────────────────────────────────────────────────


class TestIsAiKey:
    """Tests for _is_ai_key helper."""

    def test_exact_match_lowercase(self):
        assert _is_ai_key("parameters")

    def test_exact_match_mixed_case(self):
        assert _is_ai_key("Parameters")

    def test_keyword_substring(self):
        assert _is_ai_key("stable_diffusion_model_v2")

    def test_c2pa_detected(self):
        assert _is_ai_key("c2pa_chunk")

    def test_standard_key_not_flagged(self):
        assert not _is_ai_key("Author")

    def test_innocuous_key_not_flagged(self):
        assert not _is_ai_key("Title")

    def test_dpi_not_flagged(self):
        assert not _is_ai_key("dpi")


# ── has_ai_metadata / get_ai_metadata ───────────────────────────────


class TestHasAiMetadata:
    """Tests for detecting AI metadata in images."""

    def test_detects_ai_metadata(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_clean_image_no_ai(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_detects_c2pa_uuid_in_isobmff_container(self, tmp_path: Path):
        """C2PA in AVIF/HEIF/MP4 lives in a ``uuid`` box identified by a fixed UUID.

        Real AVIF/HEIF fixtures aren't shipped, so simulate the container by
        prepending an ISOBMFF-shaped ftyp box and the C2PA UUID bytes.
        """
        from remove_ai_watermarks.metadata import C2PA_UUID

        path = tmp_path / "fake.avif"
        # ftyp box: size(4) + 'ftyp' + 'avif' + minor_version(4) + 'avif'
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"jumb-payload"
        path.write_bytes(ftyp + uuid_box + b"\x00" * 64)
        assert has_ai_metadata(path)

    def test_strip_c2pa_boxes_removes_uuid_box(self, tmp_path: Path):
        """ISOBMFF strip should drop the C2PA uuid box and keep everything else."""
        from remove_ai_watermarks.metadata import C2PA_UUID
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload (8 bytes -> total 32)
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        cleaned, stripped = strip_c2pa_boxes(ftyp + uuid_box + mdat)
        assert stripped == 1
        assert cleaned == ftyp + mdat

    def test_strip_c2pa_boxes_passthrough_for_non_isobmff(self):
        """Non-ISOBMFF input must be returned unchanged."""
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
        cleaned, stripped = strip_c2pa_boxes(data)
        assert stripped == 0
        assert cleaned == data

    def test_remove_ai_metadata_strips_c2pa_in_avif(self, tmp_path: Path):
        """End-to-end: ``remove_ai_metadata`` on a fake .avif drops the C2PA box."""
        from remove_ai_watermarks.metadata import C2PA_UUID, remove_ai_metadata

        src = tmp_path / "in.avif"
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        src.write_bytes(ftyp + uuid_box + mdat)

        out = tmp_path / "out.avif"
        result = remove_ai_metadata(src, out)
        assert result == out
        assert out.read_bytes() == ftyp + mdat
        # And after stripping, detection must no longer flag the cleaned file.
        from remove_ai_watermarks.metadata import has_ai_metadata

        assert not has_ai_metadata(out)

    def test_remove_ai_metadata_blanks_exif_token_item_in_avif(self, tmp_path: Path):
        """End-to-end: ``remove_ai_metadata`` blanks an AI-generator EXIF token
        stored as a meta-box Exif item (bytes in mdat) without re-encoding."""
        from remove_ai_watermarks.metadata import remove_ai_metadata

        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        blob = piexif.dump({"0th": {piexif.ImageIFD.Software: b"Midjourney", piexif.ImageIFD.Make: b"NIKON"}})
        mdat = struct.pack(">I", 8 + len(blob)) + b"mdat" + blob
        src = tmp_path / "in.avif"
        src.write_bytes(ftyp + mdat)

        out = tmp_path / "out.avif"
        remove_ai_metadata(src, out)
        cleaned = out.read_bytes()
        assert len(cleaned) == len(ftyp + mdat)  # in place, no re-encode
        assert b"Midjourney" not in cleaned  # AI token gone
        assert b"NIKON" in cleaned  # camera tag preserved

    def test_detects_iptc_trained_algorithmic_media_marker(self, tmp_path: Path):
        """Some pipelines embed only the IPTC AI marker in XMP, no C2PA manifest."""
        path = tmp_path / "fake.jpg"
        # Minimal JPEG-ish bytes containing the IPTC AI marker in an XMP-like blob.
        xmp = (
            b"<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>"
            b"trainedAlgorithmicMedia"
            b"</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>"
        )
        path.write_bytes(b"\xff\xd8\xff\xe1" + xmp + b"\xff\xd9")
        assert has_ai_metadata(path)

    @pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
    def test_jpeg_metadata_strip_is_pixel_lossless(self, tmp_path: Path):
        """A JPEG metadata strip must NOT re-encode the DCT scan: pixels stay
        bit-identical, only the AI provenance APP segments are removed. Verified on real
        committed fixtures -- grok-1.jpg (EXIF signature), flux-1.jpg (C2PA APP11)."""
        import numpy as np

        from remove_ai_watermarks import image_io
        from remove_ai_watermarks.metadata import remove_ai_metadata

        for name in ("grok-1.jpg", "flux-1.jpg"):
            src = SAMPLES_DIR / name
            if not src.exists():
                continue
            before = image_io.imread(str(src))
            out = tmp_path / name
            remove_ai_metadata(src, out)
            after = image_io.imread(str(out))
            assert before is not None
            assert after is not None
            assert np.array_equal(before, after), f"{name}: pixels changed (DCT was re-encoded)"
            assert not has_ai_metadata(out), f"{name}: AI metadata survived the strip"

    @staticmethod
    def _xmp_iptc_jpeg(tmp_path: Path, name: str, marker: bytes) -> Path:
        """A real (decodable) JPEG carrying the IPTC AI marker in a well-formed APP1
        XMP segment -- the layout Instagram/Facebook/X and MidJourney/Meta use, where
        ``digitalSourceType`` lives in XMP rather than the APP13 IPTC-IIM record.
        Synthetic (no corpus): a solid cv2 JPEG with the APP1 spliced in after SOI."""
        import cv2
        import numpy as np

        real = tmp_path / f"real-{name}"
        cv2.imwrite(str(real), np.full((32, 32, 3), 200, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        data = real.read_bytes()
        xmp = (
            b"http://ns.adobe.com/xap/1.0/\x00"
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description>'
            b"<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/"
            + marker
            + b"</Iptc4xmpExt:DigitalSourceType></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        seg = b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp
        path = tmp_path / name
        path.write_bytes(data[:2] + seg + data[2:])  # splice APP1 right after SOI
        return path

    @pytest.mark.parametrize("marker", [b"trainedAlgorithmicMedia", b"AISystemUsed"])
    def test_jpeg_strip_removes_iptc_marker_in_xmp(self, tmp_path: Path, marker: bytes):
        """Regression: the lossless JPEG strip must drop an AI-bearing APP1 XMP packet
        when the AI signal is an IPTC ``digitalSourceType`` / 2025.1 field, not only a
        C2PA or China-AIGC token. Before the fix these survived because the APP1 branch
        of ``_jpeg_app_carries_ai`` checked only c2pa + AIGC markers, leaving the
        Instagram/MidJourney/Meta 'Made with AI' XMP intact. Pixels stay bit-identical."""
        import numpy as np

        from remove_ai_watermarks import image_io
        from remove_ai_watermarks.metadata import remove_ai_metadata

        src = self._xmp_iptc_jpeg(tmp_path, "iptc-xmp.jpg", marker)
        assert has_ai_metadata(src)  # detected before
        before = image_io.imread(str(src))
        out = tmp_path / "clean.jpg"
        remove_ai_metadata(src, out)
        after = image_io.imread(str(out))
        assert before is not None
        assert after is not None
        assert np.array_equal(before, after), "pixels changed: the DCT scan was re-encoded"
        assert not has_ai_metadata(out), "IPTC AI marker in XMP survived the strip"

    def test_remove_ai_metadata_failsafe_on_truncated_png(self, tmp_path: Path):
        """Regression: a truncated / corrupt image must NOT crash remove_ai_metadata (a
        direct library caller like a web worker would 500 on a partial upload). PIL raises
        OSError decoding a truncated PNG; the strip must fail SAFE -- copy the input through
        unchanged and return, mirroring strip_c2pa_boxes. Real prod corpus: ~0.2% of
        uploads are truncated."""
        import cv2
        import numpy as np

        from remove_ai_watermarks.metadata import remove_ai_metadata

        real = tmp_path / "real.png"
        cv2.imwrite(str(real), np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8))
        truncated = tmp_path / "truncated.png"
        truncated.write_bytes(real.read_bytes()[: real.stat().st_size // 2])  # chop the IDAT stream
        out = tmp_path / "out.png"
        # Must not raise; output is the input copied through (undecodable -> nothing to strip).
        result = remove_ai_metadata(truncated, out)
        assert result == out
        assert out.read_bytes() == truncated.read_bytes()


class TestC2paMarkerIn:
    """The C2PA presence check requires a JUMBF wrapper or the C2PA uuid box, so
    a bare 4-byte ``c2pa`` substring (e.g. random compressed pixel data) does not
    false-positive -- the regression behind 4 cleaned PNGs re-flagging C2PA."""

    def test_jumbf_wrapped_c2pa_detected(self):
        assert c2pa_marker_in(b"....jumbc2pa....manifest....") is True

    def test_c2pa_uuid_box_detected(self):
        assert c2pa_marker_in(b"\x00\x00\x00\x18uuid" + C2PA_UUID + b"payload") is True

    def test_bare_c2pa_substring_not_detected(self):
        # The exact false positive: "c2pa" appears in noise but no JUMBF/uuid box.
        assert c2pa_marker_in(b"\x9c\xc3\xa7B1\x11c2pa\x80b\x804\xc5\xf9random idat") is False

    def test_jumb_without_c2pa_not_detected(self):
        assert c2pa_marker_in(b"some jumb box but no manifest label") is False

    def test_empty_not_detected(self):
        assert c2pa_marker_in(b"") is False


class TestSamsungGenai:
    """Samsung Galaxy AI editing marker (genAIType in PhotoEditor_Re_Edit_Data).

    Synthetic byte blobs -- real Galaxy files are user content and not shipped
    (public repo), same discipline as the Grok/Doubao fixtures.
    """

    @staticmethod
    def _samsung_jpeg(tmp_path: Path, name: str, payload: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(b"\xff\xd8\xff\xe1" + payload + b"\xff\xd9")
        return path

    def test_nonzero_genai_type_detected(self, tmp_path: Path):
        p = self._samsung_jpeg(
            tmp_path, "galaxy.jpg", b'PhotoEditor_Re_Edit_Data{"connectorType":"srvg","genAIType":1}'
        )
        assert samsung_genai(p) == 1

    def test_other_nonzero_value_detected(self, tmp_path: Path):
        p = self._samsung_jpeg(tmp_path, "galaxy5.jpg", b'PhotoEditor_Re_Edit_Data{"genAIType":5}')
        assert samsung_genai(p) == 5

    def test_zero_genai_type_is_none(self, tmp_path: Path):
        """genAIType:0 means no generative AI was used -- not a positive signal."""
        p = self._samsung_jpeg(tmp_path, "edit.jpg", b'PhotoEditor_Re_Edit_Data{"genAIType":0}')
        assert samsung_genai(p) is None

    def test_genai_without_editor_container_ignored(self, tmp_path: Path):
        """An incidental genAIType token outside Samsung's editor JSON is ignored."""
        p = self._samsung_jpeg(tmp_path, "stray.jpg", b'some other blob "genAIType":1 elsewhere')
        assert samsung_genai(p) is None

    def test_remove_strips_post_eoi_trailer(self, tmp_path: Path):
        """Regression: Galaxy AI appends ``PhotoEditor_Re_Edit_Data`` as a trailer AFTER
        the JPEG EOI, so the verbatim scan copy in the lossless strip carried it through
        (survived 0/8 on the corpus). The strip now truncates a Samsung AI trailer at EOI;
        pixels stay bit-identical and a non-Samsung trailer is preserved."""
        import cv2
        import numpy as np

        from remove_ai_watermarks import image_io
        from remove_ai_watermarks.metadata import remove_ai_metadata

        real = tmp_path / "real.jpg"
        cv2.imwrite(str(real), np.full((32, 32, 3), 180, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
        src = tmp_path / "galaxy.jpg"
        src.write_bytes(real.read_bytes() + b'PhotoEditor_Re_Edit_Data{"genAIType":1}')
        assert samsung_genai(src) == 1  # detected before
        before = image_io.imread(str(src))
        out = tmp_path / "clean.jpg"
        remove_ai_metadata(src, out)
        after = image_io.imread(str(out))
        assert before is not None
        assert after is not None
        assert np.array_equal(before, after), "pixels changed: the DCT scan was re-encoded"
        assert samsung_genai(out) is None, "Samsung genAIType trailer survived the strip"

    def test_non_samsung_trailer_preserved(self, tmp_path: Path):
        """A benign post-EOI trailer (e.g. an MPF block) must NOT be truncated."""
        import cv2
        import numpy as np

        from remove_ai_watermarks.metadata import _strip_samsung_trailer

        real = tmp_path / "real.jpg"
        cv2.imwrite(str(real), np.full((16, 16, 3), 90, np.uint8))
        tail = real.read_bytes() + b"MPF-benign-trailer-bytes"
        assert _strip_samsung_trailer(tail) == tail

    def test_detects_trailer_past_scan_window(self, tmp_path: Path):
        """Regression: the marker is a trailer AFTER the JPEG EOI, so on a multi-MB
        photo it sits past the 512 KB quick-scan window. Detection must read the file
        tail too, else it disagrees with removal (which reads the whole file). A random
        1400x1400 q100 JPEG exceeds 512 KB; the marker is only in its post-EOI tail."""
        import cv2
        import numpy as np

        real = tmp_path / "big.jpg"
        big = np.random.default_rng(0).integers(0, 256, (1400, 1400, 3), dtype=np.uint8)
        cv2.imwrite(str(real), big, [cv2.IMWRITE_JPEG_QUALITY, 100])
        assert real.stat().st_size > 512 * 1024  # trailer will be past the quick-scan window
        p = tmp_path / "galaxy_big.jpg"
        p.write_bytes(real.read_bytes() + b'PhotoEditor_Re_Edit_Data{"genAIType":1}')
        assert samsung_genai(p) == 1

    def test_clean_image_is_none(self, tmp_clean_png):
        assert samsung_genai(tmp_clean_png) is None

    def test_surfaced_in_get_ai_metadata(self, tmp_path: Path):
        p = self._samsung_jpeg(tmp_path, "galaxy.jpg", b'PhotoEditor_Re_Edit_Data{"genAIType":1}')
        meta = get_ai_metadata(p)
        assert "samsung_genai" in meta
        assert "genAIType=1" in meta["samsung_genai"]


class TestGetAiMetadata:
    """Tests for extracting AI metadata."""

    def test_extracts_parameters_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta
        assert "Euler" in meta["parameters"]

    def test_extracts_prompt_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "prompt" in meta

    def test_does_not_extract_author(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "Author" not in meta

    def test_clean_image_empty_dict(self, tmp_clean_png):
        meta = get_ai_metadata(tmp_clean_png)
        assert meta == {}

    def test_long_value_is_truncated(self, tmp_path: Path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "x" * 300)
        path = tmp_path / "long.png"
        img.save(path, pnginfo=pnginfo)
        meta = get_ai_metadata(path)
        assert meta["parameters"].endswith("…")
        assert len(meta["parameters"]) <= 205

    def test_unopenable_file_does_not_raise(self, tmp_path: Path):
        # PIL can't open HEIC without pillow-heif; get_ai_metadata must fall
        # through to the binary scan, not propagate UnidentifiedImageError.
        path = tmp_path / "iphone.heic"
        path.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
        assert get_ai_metadata(path) == {}


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestGetAiMetadataRealSample:
    """get_ai_metadata surfaces the consolidated C2PA fields on real images."""

    def test_openai_sample_fields(self):
        meta = get_ai_metadata(SAMPLES_DIR / "chatgpt-1.png")
        assert "claim_generator" in meta
        assert "OpenAI" in meta["issuer"]
        assert "OpenAI" in meta["synthid_watermark"]
        assert "trainedAlgorithmicMedia" in meta["source_type"]


@pytest.mark.parametrize(
    "marker",
    [
        b"trainedAlgorithmicMedia",
        b"compositeSynthetic",
        b"compositeWithTrainedAlgorithmicMedia",
    ],
)
def test_has_ai_metadata_detects_each_iptc_marker(tmp_path: Path, marker: bytes):
    """Each IPTC digitalSourceType AI marker in XMP triggers detection."""
    path = tmp_path / "iptc.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe1<x:xmpmeta>" + marker + b"</x:xmpmeta>\xff\xd9")
    assert has_ai_metadata(path)


def test_bare_algorithmic_media_not_flagged_ai(tmp_path: Path):
    """Regression: the IPTC ``algorithmicMedia`` digitalSourceType is PROCEDURAL (an
    algorithm not trained on sampled data), NOT AI/ML generation. It must NOT be flagged
    -- flagging it made identify assert is_ai=high + has_invisible_target=True, which
    would trigger a diffusion scrub of clean procedural content. It is a distinct token
    from ``trainedAlgorithmicMedia``, so real 'Made with AI' labels are unaffected."""
    path = tmp_path / "proc.jpg"
    path.write_bytes(
        b"\xff\xd8\xff\xe1<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>"
        b"http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia"
        b"</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>\xff\xd9"
    )
    assert not has_ai_metadata(path)


# ── SynthID-source detection (metadata proxy) ────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestSynthIDSource:
    """SynthID detection via the C2PA companion manifest.

    Google (Imagen/Gemini) and OpenAI (ChatGPT/DALL-E/gpt-image) pair an
    invisible SynthID pixel watermark with a C2PA manifest. Adobe Firefly and
    Microsoft Designer sign C2PA Content Credentials but do NOT use SynthID,
    so the discriminating signal is the C2PA *issuer*, not the mere presence
    of a manifest. These tests run against real, committed sample images.
    """

    def test_openai_chatgpt_is_synthid_source(self):
        assert synthid_source(SAMPLES_DIR / "chatgpt-1.png") == "OpenAI"

    def test_openai_verdict_in_get_ai_metadata(self):
        meta = get_ai_metadata(SAMPLES_DIR / "chatgpt-1.png")
        assert "synthid_watermark" in meta
        assert "OpenAI" in meta["synthid_watermark"]

    def test_adobe_firefly_is_not_synthid_source(self):
        # Adobe signs C2PA (trainedAlgorithmicMedia) but embeds no SynthID.
        assert synthid_source(SAMPLES_DIR / "firefly-1.png") is None
        assert "synthid_watermark" not in get_ai_metadata(SAMPLES_DIR / "firefly-1.png")

    def test_non_ai_image_is_not_synthid_source(self, clean_photo: Path):
        assert synthid_source(clean_photo) is None


class TestSynthIDSourceNonPng:
    """SynthID-source detection must work beyond PNG.

    ChatGPT/Gemini images saved as JPEG/WebP/AVIF carry their C2PA manifest in
    a non-PNG container (JPEG APP11, ISOBMFF uuid box), so the PNG caBX parser
    misses them. These use synthetic byte blobs (real fixtures aren't shipped).
    """

    def _c2pa_jpeg(self, tmp_path: Path, name: str, issuer: bytes, marker: bytes = b"trainedAlgorithmicMedia") -> Path:
        path = tmp_path / name
        # Minimal JPEG shell with an embedded C2PA-ish blob.
        blob = b"jumbc2pa" + issuer + b"..." + marker
        path.write_bytes(b"\xff\xd8\xff\xe1" + blob + b"\xff\xd9")
        return path

    def test_openai_c2pa_in_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, "chatgpt.jpg", b"OpenAI")
        assert synthid_source(path) == "OpenAI"

    def test_google_c2pa_in_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, "gemini.jpg", b"Google")
        assert synthid_source(path) == "Google LLC"

    def test_adobe_c2pa_in_jpeg_is_none(self, tmp_path: Path):
        # Adobe signs C2PA but embeds no SynthID.
        path = self._c2pa_jpeg(tmp_path, "firefly.jpg", b"Adobe")
        assert synthid_source(path) is None

    def test_openai_without_ai_marker_is_none(self, tmp_path: Path):
        # Issuer present but no AI digital-source marker -> not a SynthID source.
        path = self._c2pa_jpeg(tmp_path, "edited.jpg", b"OpenAI", marker=b"")
        assert synthid_source(path) is None


# ── remove_ai_metadata ──────────────────────────────────────────────


class TestRemoveAiMetadata:
    """Tests for stripping AI metadata."""

    def test_removes_ai_keys(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output)

        with Image.open(output) as img:
            assert "parameters" not in img.info
            assert "prompt" not in img.info

    def test_keeps_standard_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=True)

        with Image.open(output) as img:
            assert "Author" in img.info
            assert img.info["Author"] == "Test Author"

    def test_remove_all_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=False)
        with Image.open(output) as img:
            assert "Author" not in img.info
            assert "parameters" not in img.info

    def test_overwrite_in_place(self, tmp_path):
        """When output_path is None, should overwrite source."""
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test data")
        path = tmp_path / "inplace.png"
        img.save(path, pnginfo=pnginfo)

        result = remove_ai_metadata(path)
        assert result == path

        with Image.open(path) as cleaned:
            assert "parameters" not in cleaned.info

    def test_jpeg_output(self, tmp_path):
        """Test metadata removal for JPEG format."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test")
        png_path = tmp_path / "source.png"
        img.save(png_path, pnginfo=pnginfo)

        jpg_path = tmp_path / "output.jpg"
        result = remove_ai_metadata(png_path, jpg_path)
        assert result == jpg_path
        assert jpg_path.exists()

    def test_jpeg_output_is_high_quality(self, tmp_path):
        """JPEG output uses high quality + 4:4:4 (no chroma subsampling), not the
        lossy PIL defaults (quality 75, 4:2:0) that visibly degrade the image."""
        from PIL.JpegImagePlugin import get_sampling

        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        png_path = tmp_path / "source.png"
        img.save(png_path)

        jpg_path = tmp_path / "output.jpg"
        remove_ai_metadata(png_path, jpg_path)

        with Image.open(jpg_path) as out:
            assert get_sampling(out) == 0  # 4:4:4, no chroma subsampling
            # quality 95 quantization tables stay well below the q75 defaults
            # (whose max quant value is ~40+); q95 tops out around 12.
            assert max(max(t) for t in out.quantization.values()) <= 15

    def test_webp_output_preserves_format_losslessly(self, tmp_path):
        """A .webp output keeps the WebP format (not silently rewritten to PNG)
        and is pixel-identical to the source (lossless)."""
        import numpy as np

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        src = Image.fromarray(arr, "RGB")
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "ai stuff")
        png_path = tmp_path / "source.png"
        src.save(png_path, pnginfo=pnginfo)

        webp_path = tmp_path / "output.webp"
        remove_ai_metadata(png_path, webp_path)

        with Image.open(webp_path) as out:
            assert out.format == "WEBP"
            assert np.array_equal(np.asarray(out.convert("RGB")), arr)

    def test_creates_parent_directories(self, tmp_path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", "test")
        path = tmp_path / "source.png"
        img.save(path, pnginfo=pnginfo)

        output = tmp_path / "sub" / "dir" / "cleaned.png"
        remove_ai_metadata(path, output)
        assert output.exists()

    def test_returns_path(self, tmp_clean_png):
        output = tmp_clean_png.parent / "out.png"
        result = remove_ai_metadata(tmp_clean_png, output)
        assert isinstance(result, Path)
        assert result == output

    def _sd_png(self, tmp_path: Path) -> Path:
        img = Image.new("RGB", (32, 32), color=(80, 80, 80))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "Steps: 20, Sampler: Euler")
        img.save(tmp_path / "sd.png", pnginfo=pnginfo)
        return tmp_path / "sd.png"

    def test_png_to_jpeg_strips_ai(self, tmp_path):
        # Cross-format output: the AI text chunk must not survive the PNG->JPEG
        # re-encode, by detection AND by raw bytes.
        out = tmp_path / "clean.jpg"
        remove_ai_metadata(self._sd_png(tmp_path), out)
        assert not has_ai_metadata(out)
        body = out.read_bytes()
        assert b"parameters" not in body
        assert b"Steps" not in body

    def test_png_to_webp_strips_ai(self, tmp_path):
        out = tmp_path / "clean.webp"
        remove_ai_metadata(self._sd_png(tmp_path), out)
        assert not has_ai_metadata(out)
        body = out.read_bytes()
        assert b"parameters" not in body
        assert b"Steps" not in body


def _img_with_software(tmp_path: Path, fmt: str, software: str) -> Path:
    """Write a tiny image carrying an EXIF Software tag."""
    exif = piexif.dump({"0th": {piexif.ImageIFD.Software: software.encode()}, "Exif": {}, "GPS": {}, "1st": {}})
    path = tmp_path / f"img.{fmt}"
    Image.new("RGB", (64, 64), (100, 90, 80)).save(path, exif=exif)
    return path


class TestExifGenerator:
    """exif_generator extracts AI-tool names from EXIF/XMP across formats."""

    def test_avif_software_ai_tool_detected(self, tmp_path: Path):
        path = _img_with_software(tmp_path, "avif", "Adobe Firefly")
        assert exif_generator(path) == "Adobe Firefly"

    def test_jpeg_software_ai_tool_detected(self, tmp_path: Path):
        path = _img_with_software(tmp_path, "jpg", "ComfyUI v1.2")
        result = exif_generator(path)
        assert result is not None
        assert "ComfyUI" in result

    def test_plain_editor_not_flagged(self, tmp_path: Path):
        # An ordinary editor tag carries no AI token and must not be flagged.
        path = _img_with_software(tmp_path, "jpg", "Adobe Photoshop 25.0")
        assert exif_generator(path) is None

    def test_make_tag_ai_tool_detected(self, tmp_path: Path):
        # Ideogram tags its output with EXIF Make="Ideogram AI" (verified on a
        # real download), so the Make tag must be read too.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Ideogram AI"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "ideogram.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) == "Ideogram AI"

    def test_camera_make_not_flagged(self, tmp_path: Path):
        # A real camera Make ("Apple") carries no AI token -> not flagged.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Apple"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "iphone.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) is None

    def test_artist_tag_ai_tool_detected(self, tmp_path: Path):
        # exif_generator also reads the EXIF Artist field for an AI token.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Artist: b"Midjourney"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "artist.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) == "Midjourney"

    def test_novelai_png_text_chunk_detected(self, tmp_path: Path):
        # NovelAI (mined corpus) stamps its generator in PNG tEXt Software/Source/
        # Title chunks, not EXIF -- the PNG-text path must catch it.
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        info.add_text("Software", "NovelAI")
        info.add_text("Source", "NovelAI Diffusion V4.5 C02D4F98")
        info.add_text("Title", "NovelAI generated image")
        path = tmp_path / "novelai.png"
        Image.new("RGB", (64, 64)).save(path, pnginfo=info)
        assert exif_generator(path) == "NovelAI"

    def test_reve_software_detected(self, tmp_path: Path):
        # Reve Image (mined corpus) writes EXIF Software="reve.com".
        path = _img_with_software(tmp_path, "jpg", "reve.com")
        assert exif_generator(path) == "reve.com"

    def test_reve_token_not_overmatched(self, tmp_path: Path):
        # The "reve.com" token must not fire on incidental words like "forever".
        path = _img_with_software(tmp_path, "jpg", "Forever Editor 2.0")
        assert exif_generator(path) is None

    def test_aphrodite_make_detected(self, tmp_path: Path):
        # Aphrodite AI (mined corpus) writes EXIF Make="Aphrodite AI".
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Aphrodite AI"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "aphrodite.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert exif_generator(path) == "Aphrodite AI"

    def test_novelai_removal_parity(self, tmp_path: Path):
        # Detection and removal must stay in parity: NovelAI stamps Title/Source
        # under non-AI keys (AI-shaped VALUE), so removal must drop them by value,
        # not only by key -- else the cleaned file still reads as NovelAI.
        from PIL.PngImagePlugin import PngInfo

        from remove_ai_watermarks.metadata import remove_ai_metadata

        info = PngInfo()
        info.add_text("Software", "NovelAI")
        info.add_text("Source", "NovelAI Diffusion V4.5 C02D4F98")
        info.add_text("Title", "NovelAI generated image")
        src = tmp_path / "novelai.png"
        Image.new("RGB", (64, 64)).save(src, pnginfo=info)
        assert exif_generator(src) == "NovelAI"

        out = tmp_path / "clean.png"
        remove_ai_metadata(src, out)
        assert exif_generator(out) is None

    def test_imagedescription_tag_ai_tool_detected(self, tmp_path: Path):
        # ...and the EXIF ImageDescription field.
        exif = piexif.dump(
            {"0th": {piexif.ImageIFD.ImageDescription: b"Made with Stable Diffusion"}, "Exif": {}, "GPS": {}, "1st": {}}
        )
        path = tmp_path / "desc.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        result = exif_generator(path)
        assert result is not None
        assert "Stable Diffusion" in result

    def test_xmp_creatortool_scan_covers_unopenable(self, tmp_path: Path):
        # PIL can't open this fake HEIF; the raw XMP CreatorTool scan still works.
        path = tmp_path / "fake.heic"
        path.write_bytes(
            b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"
            b"<x:xmpmeta><xmp:CreatorTool>Midjourney v7</xmp:CreatorTool></x:xmpmeta>"
        )
        result = exif_generator(path)
        assert result is not None
        assert "Midjourney" in result

    def test_clean_image_is_none(self, tmp_clean_png: Path):
        assert exif_generator(tmp_clean_png) is None


_FAKE_SIG = "A" * 120  # 64+ base64 chars; real Grok payloads are 300-1004
_FAKE_UUID = "12345678-1234-1234-1234-123456789abc"


def _grok_jpeg(tmp_path: Path, *, signature: str = _FAKE_SIG, artist: str = _FAKE_UUID) -> Path:
    """Write a synthetic Grok-style JPEG: EXIF ImageDescription "Signature: ..."
    + a UUID Artist. Synthetic on purpose -- never commit a real Grok image
    (its Artist UUID + signature are user/session data; this is a public repo)."""
    exif = piexif.dump(
        {
            "0th": {
                piexif.ImageIFD.ImageDescription: f"Signature: {signature}".encode("latin1"),
                piexif.ImageIFD.Artist: artist.encode("latin1"),
            },
            "Exif": {},
            "GPS": {},
            "1st": {},
        }
    )
    path = tmp_path / "grok.jpg"
    Image.new("RGB", (64, 64), (70, 80, 90)).save(path, exif=exif)
    return path


class TestXaiSignature:
    """xAI / Grok's EXIF Signature + UUID-Artist provenance scheme."""

    def test_signature_plus_uuid_detected(self, tmp_path: Path):
        assert xai_signature(_grok_jpeg(tmp_path)) is True

    def test_real_grok_sample_detected(self):
        # Real committed Grok download (data/samples/grok-1.jpg); the EXIF
        # Signature + UUID-Artist pair is the only AI signal it carries.
        assert xai_signature(SAMPLES_DIR / "grok-1.jpg") is True

    def test_signature_without_uuid_artist_not_flagged(self, tmp_path: Path):
        # A "Signature:" blob but a non-UUID Artist is not the Grok pair.
        assert xai_signature(_grok_jpeg(tmp_path, artist="John Doe")) is False

    def test_bare_uuid_artist_not_flagged(self, tmp_path: Path):
        # A UUID Artist alone (no Signature blob) must not false-positive.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Artist: _FAKE_UUID.encode()}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "uuid_only.jpg"
        Image.new("RGB", (64, 64)).save(path, exif=exif)
        assert xai_signature(path) is False

    def test_short_signature_text_not_flagged(self, tmp_path: Path):
        # Incidental short "Signature: ..." text is below the 64-char base64 bar.
        assert xai_signature(_grok_jpeg(tmp_path, signature="ok")) is False

    def test_clean_image_is_false(self, tmp_clean_png: Path):
        assert xai_signature(tmp_clean_png) is False

    def test_surfaced_in_get_ai_metadata(self, tmp_path: Path):
        assert "xai_signature" in get_ai_metadata(_grok_jpeg(tmp_path))

    def test_has_ai_metadata_true(self, tmp_path: Path):
        assert has_ai_metadata(_grok_jpeg(tmp_path)) is True


class TestRemoveAiExif:
    """remove_ai_metadata scrubs AI-provenance EXIF tags but keeps genuine EXIF."""

    def test_grok_signature_stripped_on_jpeg_output(self, tmp_path: Path):
        src = _grok_jpeg(tmp_path)
        assert xai_signature(src) is True
        out = tmp_path / "clean.jpg"
        remove_ai_metadata(src, out)
        assert xai_signature(out) is False
        assert has_ai_metadata(out) is False

    def test_generator_make_token_stripped(self, tmp_path: Path):
        # Ideogram's EXIF Make="Ideogram AI" must be scrubbed on removal.
        exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Ideogram AI"}, "Exif": {}, "GPS": {}, "1st": {}})
        src = tmp_path / "ideogram.jpg"
        Image.new("RGB", (64, 64)).save(src, exif=exif)
        out = tmp_path / "clean.jpg"
        remove_ai_metadata(src, out)
        assert exif_generator(out) is None

    def test_real_camera_exif_preserved(self, tmp_path: Path):
        # A real-camera Make ("Apple") carries no AI token and must survive.
        exif = piexif.dump(
            {
                "0th": {piexif.ImageIFD.Make: b"Apple", piexif.ImageIFD.Model: b"iPhone 15"},
                "Exif": {},
                "GPS": {},
                "1st": {},
            }
        )
        src = tmp_path / "photo.jpg"
        Image.new("RGB", (64, 64)).save(src, exif=exif)
        out = tmp_path / "out.jpg"
        remove_ai_metadata(src, out)
        kept = piexif.load(Image.open(out).info["exif"])["0th"]
        assert kept.get(piexif.ImageIFD.Make) == b"Apple"

    def test_xai_pair_stripped_but_genuine_camera_tags_kept(self, tmp_path: Path):
        # An image carrying BOTH the xAI Signature pair (ImageDescription =
        # "Signature: <base64>" + UUID Artist) AND genuine non-AI camera tags.
        # The scrub must delete only the xAI pair, leaving the camera tags intact.
        sig = "Signature: " + "A" * 120
        artist = "12345678-1234-1234-1234-123456789abc"
        exif = piexif.dump(
            {
                "0th": {
                    piexif.ImageIFD.ImageDescription: sig.encode(),
                    piexif.ImageIFD.Artist: artist.encode(),
                    piexif.ImageIFD.Make: b"Canon",
                    piexif.ImageIFD.Model: b"EOS R5",
                },
                "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2024:01:01 12:00:00"},
                "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N"},
                "1st": {},
            }
        )
        src = tmp_path / "grok_plus_cam.jpg"
        Image.new("RGB", (32, 32)).save(src, exif=exif)
        out = tmp_path / "scrubbed.jpg"
        remove_ai_metadata(src, out)

        # xAI signature pair is gone (xai_signature returns a bool, not None).
        assert xai_signature(out) is False
        kept = piexif.load(Image.open(out).info["exif"])
        assert kept["0th"].get(piexif.ImageIFD.ImageDescription) is None
        assert kept["0th"].get(piexif.ImageIFD.Artist) is None
        # Genuine camera tags are preserved.
        assert kept["0th"].get(piexif.ImageIFD.Make) == b"Canon"
        assert kept["0th"].get(piexif.ImageIFD.Model) == b"EOS R5"
        assert kept["Exif"].get(piexif.ExifIFD.DateTimeOriginal) == b"2024:01:01 12:00:00"
        assert kept["GPS"].get(piexif.GPSIFD.GPSLatitudeRef) == b"N"


class TestAIGCLabel:
    """China TC260 AIGC labeling (Doubao and other China-served generators)."""

    def _aigc_png(self, tmp_path: Path, label: str = "1", producer: str = "TESTPRODUCER001") -> Path:
        from remove_ai_watermarks.metadata import aigc_label  # noqa: F401  (import-time guard)

        p = tmp_path / "doubao.png"
        Image.new("RGB", (32, 32)).save(p)
        # XMP is HTML-entity encoded in real files; aigc_label must unescape it.
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" '
            'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"><TC260:AIGC>'
            f"{{&quot;Label&quot;:&quot;{label}&quot;,&quot;ContentProducer&quot;:&quot;{producer}&quot;}}"
            "</TC260:AIGC></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        with open(p, "ab") as f:
            f.write(xmp.encode())
        return p

    def test_parses_label_and_producer(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_png(tmp_path))
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"] == "TESTPRODUCER001"

    def test_none_when_absent(self, tmp_clean_png):
        from remove_ai_watermarks.metadata import aigc_label

        assert aigc_label(tmp_clean_png) is None

    def test_has_ai_metadata_detects_aigc(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_png(tmp_path))

    def test_get_ai_metadata_surfaces_aigc(self, tmp_path: Path):
        meta = get_ai_metadata(self._aigc_png(tmp_path))
        assert "aigc_label" in meta
        assert "TC260" in meta["aigc_label"]

    def _aigc_chunk_png(self, tmp_path: Path, producer: str = "doubao") -> Path:
        """Doubao writes the TC260 object as a PNG ``tEXt`` chunk keyed ``AIGC``
        with raw JSON (no XMP, no namespaced marker)."""
        import json

        p = tmp_path / "doubao_chunk.png"
        pnginfo = PngInfo()
        pnginfo.add_text(
            "AIGC",
            json.dumps({"Label": "1", "ContentProducer": producer, "ProduceID": "abc123"}),
        )
        Image.new("RGB", (32, 32)).save(p, pnginfo=pnginfo)
        return p

    def test_parses_png_text_chunk_form(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_chunk_png(tmp_path))
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"] == "doubao"

    def test_png_chunk_without_tc260_field_ignored(self, tmp_path: Path):
        """A generic ``AIGC`` chunk with no TC260 field must not false-positive."""
        import json

        from remove_ai_watermarks.metadata import aigc_label

        p = tmp_path / "unrelated.png"
        pnginfo = PngInfo()
        pnginfo.add_text("AIGC", json.dumps({"unrelated": "value"}))
        Image.new("RGB", (32, 32)).save(p, pnginfo=pnginfo)
        assert aigc_label(p) is None

    def test_has_ai_metadata_detects_png_chunk_form(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_chunk_png(tmp_path))

    def _aigc_serviceprovider_png(self, tmp_path: Path, provider: str = "腾讯云") -> Path:
        """Tencent Cloud's AIGC variant uses a service-provider schema
        (``ServiceProvider`` / ``ServiceUser`` / ``Time`` / ``ContentId``) rather
        than the producer schema, embedded in EXIF ``ImageDescription`` -- none of
        its fields overlap the producer-side ``_TC260_FIELDS``, so the generic
        ``"AIGC":{...}`` gate must recognize them too."""
        import json

        import piexif

        p = tmp_path / "tencent_aigc.png"
        payload = json.dumps(
            {
                "AIGC": {
                    "ServiceProvider": provider,
                    "ServiceUser": "1379431822",
                    "Time": "2026-06-30 23:54:50.067",
                    "ContentId": "a284dd3d-15ae-4cce-b2dd-010be84921df",
                }
            },
            ensure_ascii=False,
        )
        exif = piexif.dump(
            {"0th": {piexif.ImageIFD.ImageDescription: payload.encode("utf-8")}, "Exif": {}, "GPS": {}, "1st": {}}
        )
        Image.new("RGB", (32, 32)).save(p, exif=exif)
        return p

    def test_parses_serviceprovider_schema(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_serviceprovider_png(tmp_path))
        assert info is not None
        assert "ServiceProvider" in info

    def test_has_ai_metadata_detects_serviceprovider_schema(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_serviceprovider_png(tmp_path))

    def test_remove_strips_png_chunk_form(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label, remove_ai_metadata

        out = tmp_path / "clean.png"
        remove_ai_metadata(self._aigc_chunk_png(tmp_path), out)
        assert aigc_label(out) is None
        assert not has_ai_metadata(out)

    def _aigc_exif_jpeg(self, tmp_path: Path, producer: str = "001191440300708461136T1308L") -> Path:
        """Some China-served generators embed the raw-JSON ``{"AIGC":{...}}``
        block in JPEG EXIF (UserComment) -- no PNG chunk, no namespaced XMP."""
        import json

        import piexif

        p = tmp_path / "aigc_exif.jpg"
        Image.new("RGB", (32, 32)).save(p)
        payload = json.dumps({"AIGC": {"Label": "1", "ContentProducer": producer, "ProduceID": "abc123"}})
        exif = {"Exif": {piexif.ExifIFD.UserComment: payload.encode("ascii")}}
        piexif.insert(piexif.dump(exif), str(p))
        return p

    def test_parses_raw_json_exif_form(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_exif_jpeg(tmp_path))
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"] == "001191440300708461136T1308L"

    def test_has_ai_metadata_detects_raw_json_exif_form(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_exif_jpeg(tmp_path))

    def test_remove_strips_raw_json_exif_form(self, tmp_path: Path):
        """Regression: the TC260 AIGC ``{"AIGC":{...}}`` block Doubao embeds in EXIF
        UserComment must be scrubbed on removal. Before the fix it survived because
        ``_scrub_ai_exif`` only touched Software/Make/Artist/ImageDescription in the
        0th IFD, never UserComment in the Exif sub-IFD."""
        from remove_ai_watermarks.metadata import aigc_label, remove_ai_metadata

        out = tmp_path / "clean.jpg"
        remove_ai_metadata(self._aigc_exif_jpeg(tmp_path), out)
        assert aigc_label(out) is None
        assert not has_ai_metadata(out)

    def _aigc_bare_jpeg(self, tmp_path: Path, producer: str = "00119144030008867405X210002") -> Path:
        """Some China-served generators glue the TC260 label straight to its JSON
        as a bare ``AIGC{...}`` blob inside a JPEG APP segment (no ``"AIGC":``
        key wrapper, no PNG chunk, no namespaced XMP) -- seen near the JFIF
        header on real 2026-06 downloads."""
        p = tmp_path / "aigc_bare.jpg"
        Image.new("RGB", (32, 32)).save(p)
        raw = p.read_bytes()
        blob = b'AIGC{"Label":"1","ContentProducer":"' + producer.encode() + b'","ProduceID":"8F995586"}'
        segment = b"\xff\xe9" + (len(blob) + 2).to_bytes(2, "big") + blob  # APP9
        p.write_bytes(raw[:2] + segment + raw[2:])  # splice after SOI
        return p

    def test_parses_bare_aigc_jpeg_segment_form(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_bare_jpeg(tmp_path))
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"] == "00119144030008867405X210002"

    def test_has_ai_metadata_detects_bare_aigc_jpeg_form(self, tmp_path: Path):
        assert has_ai_metadata(self._aigc_bare_jpeg(tmp_path))

    def test_remove_strips_bare_aigc_jpeg_form(self, tmp_path: Path):
        """Regression: a bare ``AIGC{...}`` blob in a non-standard JPEG APP segment
        (APP9 here) is detected by aigc_label, so removal must drop that segment too.
        Before the fix ``_jpeg_app_carries_ai`` only inspected APP11/APP1-XMP/APP13, so
        the blob survived the lossless strip (detection<->removal parity break)."""
        from remove_ai_watermarks.metadata import aigc_label, remove_ai_metadata

        out = tmp_path / "clean.jpg"
        remove_ai_metadata(self._aigc_bare_jpeg(tmp_path), out)
        assert aigc_label(out) is None
        assert not has_ai_metadata(out)

    def test_bare_aigc_without_tc260_field_ignored(self, tmp_path: Path):
        """A bare ``AIGC{...}`` blob with no TC260 field must not false-positive."""
        from remove_ai_watermarks.metadata import aigc_label

        p = tmp_path / "bare_unrelated.jpg"
        Image.new("RGB", (32, 32)).save(p)
        raw = p.read_bytes()
        blob = b'AIGC{"unrelated":"value"}'
        segment = b"\xff\xe9" + (len(blob) + 2).to_bytes(2, "big") + blob
        p.write_bytes(raw[:2] + segment + raw[2:])
        assert aigc_label(p) is None

    def test_raw_json_without_tc260_field_ignored(self, tmp_path: Path):
        """A bare ``{"AIGC":{...}}`` object with no TC260 field must not fire."""
        import json

        import piexif

        from remove_ai_watermarks.metadata import aigc_label

        p = tmp_path / "unrelated.jpg"
        Image.new("RGB", (32, 32)).save(p)
        payload = json.dumps({"AIGC": {"unrelated": "value"}})
        exif = {"Exif": {piexif.ExifIFD.UserComment: payload.encode("ascii")}}
        piexif.insert(piexif.dump(exif), str(p))
        assert aigc_label(p) is None

    def _aigc_attr_png(self, tmp_path: Path, producer: str = "picwish") -> Path:
        """PicWish writes the TC260 label as an XMP *attribute*
        (``TC260:AIGC="{...}"``), not the nested element form."""
        p = tmp_path / "picwish.png"
        Image.new("RGB", (32, 32)).save(p)
        xmp = (
            '<rdf:Description rdf:about="" '
            'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/" '
            f'TC260:AIGC="{{&quot;Label&quot;:&quot;1&quot;,&quot;ContentProducer&quot;:&quot;{producer}&quot;}}"/>'
        )
        with open(p, "ab") as f:
            f.write(xmp.encode())
        return p

    def test_parses_xmp_attribute_form(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(self._aigc_attr_png(tmp_path))
        assert info is not None
        assert info["ContentProducer"] == "picwish"

    def test_scan_head_collects_png_metadata_past_window(self, tmp_path: Path):
        """A PNG metadata chunk beyond the read window is still reachable -- the
        regression for a TC260 XMP packet appended after a large IDAT."""
        import json

        from remove_ai_watermarks.metadata import _png_late_metadata, scan_head

        p = tmp_path / "late.png"
        pnginfo = PngInfo()
        pnginfo.add_text("AIGC", json.dumps({"Label": "1", "ContentProducer": "doubao"}))
        Image.new("RGB", (16, 16)).save(p, pnginfo=pnginfo)
        # window = 8 (just the signature) makes the text chunk "late".
        assert b"ContentProducer" in _png_late_metadata(p, 8)
        assert b"ContentProducer" in scan_head(p, 8)


class TestHuggingFaceJob:
    """HuggingFace-hosted job marker (``hf-job-id`` PNG text chunk)."""

    def _hf_png(self, tmp_path: Path, job_id: str = "ec8380a6-2091-423a-b835-209420f99ee1") -> Path:
        p = tmp_path / "hfjob.png"
        pnginfo = PngInfo()
        pnginfo.add_text("hf-job-id", job_id)
        Image.new("RGB", (32, 32)).save(p, pnginfo=pnginfo)
        return p

    def test_returns_job_id(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import huggingface_job

        assert huggingface_job(self._hf_png(tmp_path)) == "ec8380a6-2091-423a-b835-209420f99ee1"

    def test_none_when_absent(self, tmp_clean_png):
        from remove_ai_watermarks.metadata import huggingface_job

        assert huggingface_job(tmp_clean_png) is None

    def test_has_ai_metadata_detects_hf_job(self, tmp_path: Path):
        assert has_ai_metadata(self._hf_png(tmp_path))

    def test_get_ai_metadata_surfaces_hf_job(self, tmp_path: Path):
        meta = get_ai_metadata(self._hf_png(tmp_path))
        assert "huggingface_job" in meta
        assert "ec8380a6" in meta["huggingface_job"]

    def test_remove_strips_hf_job(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import huggingface_job, remove_ai_metadata

        out = tmp_path / "clean.png"
        remove_ai_metadata(self._hf_png(tmp_path), out)
        assert huggingface_job(out) is None
        assert not has_ai_metadata(out)


@pytest.mark.skipif(not (SAMPLES_DIR / "doubao-1.png").exists(), reason="doubao sample not present")
class TestAIGCRealSample:
    """Real Doubao (ByteDance) sample carries the China TC260 AIGC XMP label."""

    def test_doubao_aigc_label(self):
        from remove_ai_watermarks.metadata import aigc_label

        info = aigc_label(SAMPLES_DIR / "doubao-1.png")
        assert info is not None
        assert info["Label"] == "1"
        assert info["ContentProducer"]  # ByteDance producer code present

    def test_doubao_detected_as_ai(self):
        assert has_ai_metadata(SAMPLES_DIR / "doubao-1.png")
        assert "aigc_label" in get_ai_metadata(SAMPLES_DIR / "doubao-1.png")


class TestSoftBinding:
    """C2PA soft-binding alg identifier -> forensic-watermark vendor name."""

    def test_vendors_in_recognizes_known_algs(self):
        from remove_ai_watermarks.noai.c2pa import soft_binding_vendors_in

        assert soft_binding_vendors_in(b"...alg...com.adobe.trustmark.P...") == ["Adobe TrustMark"]
        assert soft_binding_vendors_in(b"com.digimarc.validate.1") == ["Digimarc"]
        assert soft_binding_vendors_in(b"ai.steg.api blah") == ["Steg.AI"]
        # Registry-verified vendors added in v0.6.x.
        assert soft_binding_vendors_in(b"ai.trufo.gen1.image") == ["Trufo"]
        assert soft_binding_vendors_in(b"io.iscc.v0") == ["ISCC (content code)"]

    def test_vendors_in_empty_when_absent(self):
        from remove_ai_watermarks.noai.c2pa import soft_binding_vendors_in

        assert soft_binding_vendors_in(b"no soft binding here") == []

    def test_get_ai_metadata_surfaces_soft_binding(self, tmp_path: Path):
        # Non-PNG binary-scan path: a manifest naming a soft-binding vendor.
        p = tmp_path / "fake.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe1 c2pa jumb com.adobe.trustmark.P \xff\xd9")
        assert get_ai_metadata(p).get("soft_binding") == "Adobe TrustMark"


class TestIptcAiFields:
    """IPTC 2025.1 AI-disclosure XMP properties (Iptc4xmpExt:AISystemUsed etc.)."""

    def test_detects_ai_system_used_element_form(self, tmp_path: Path):
        p = tmp_path / "iptc_ai.jpg"
        p.write_bytes(
            b"\xff\xd8\xff\xe1<x:xmpmeta><Iptc4xmpExt:AISystemUsed>ChatGPT DALL-E"
            b"</Iptc4xmpExt:AISystemUsed></x:xmpmeta>\xff\xd9"
        )
        assert has_ai_metadata(p) is True
        assert iptc_ai_system(p) == "ChatGPT DALL-E"
        assert "ChatGPT DALL-E" in get_ai_metadata(p)["ai_system"]

    def test_attribute_serialization(self, tmp_path: Path):
        p = tmp_path / "attr.jpg"
        p.write_bytes(b'\xff\xd8\xff\xe1 Iptc4xmpExt:AISystemUsed="Google Gemini" \xff\xd9')
        assert iptc_ai_system(p) == "Google Gemini"

    def test_present_without_value(self, tmp_path: Path):
        # A disclosure field with no extractable value still flags presence.
        p = tmp_path / "novalue.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe1 Iptc4xmpExt:AIPromptWriterName \xff\xd9")
        assert iptc_ai_system(p) == "fields present"
        assert has_ai_metadata(p) is True

    def test_clean_image_none(self, tmp_clean_png: Path):
        assert iptc_ai_system(tmp_clean_png) is None


# Synthetic MP4 (ISOBMFF): ftyp + C2PA uuid box + mdat. Same box format as AVIF.
_MP4_FTYP = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
_MP4_MDAT = b"\x00\x00\x00\x10mdat" + b"videodat"


def _box(box_type: bytes, payload: bytes) -> bytes:
    """Build a 32-bit-size ISOBMFF box: [size:4][type:4][payload]."""
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


class TestVideoC2pa:
    """C2PA in MP4 (ISOBMFF) -- detect + strip, reusing the image box walker."""

    def test_detects_c2pa_in_mp4(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import C2PA_UUID

        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"manifest"
        p = tmp_path / "ai.mp4"
        p.write_bytes(_MP4_FTYP + uuid_box + _MP4_MDAT)
        assert has_ai_metadata(p) is True

    def test_strips_c2pa_in_mp4(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import C2PA_UUID

        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"manifest"
        src = tmp_path / "in.mp4"
        src.write_bytes(_MP4_FTYP + uuid_box + _MP4_MDAT)
        out = tmp_path / "out.mp4"
        remove_ai_metadata(src, out)
        assert out.read_bytes() == _MP4_FTYP + _MP4_MDAT
        assert has_ai_metadata(out) is False


class TestLateProvenanceBox:
    """A C2PA / provenance box placed AFTER a large mdat (streaming / non-faststart
    MP4) must still be detected -- the fixed first-MB scan would miss it."""

    def _mp4_late_c2pa(self, tmp_path: Path, gap: int = 1_500_000) -> Path:
        from remove_ai_watermarks.metadata import C2PA_UUID

        big_mdat = _box(b"mdat", b"\x00" * gap)  # > 1 MB pushes the manifest past the scan window
        manifest = C2PA_UUID + b"OpenAI jumbf c2pa ... trainedAlgorithmicMedia ..."
        p = tmp_path / "stream.mp4"
        p.write_bytes(_MP4_FTYP + big_mdat + _box(b"uuid", manifest))
        return p

    def test_scan_c2pa_region_finds_late_box(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import C2PA_UUID
        from remove_ai_watermarks.noai.isobmff import scan_c2pa_region

        region = scan_c2pa_region(self._mp4_late_c2pa(tmp_path))
        assert C2PA_UUID in region
        assert b"trainedAlgorithmicMedia" in region

    def test_fixed_window_would_have_missed_it(self, tmp_path: Path):
        # Documents the regression the box walk fixes: the manifest is beyond 1 MB.
        from remove_ai_watermarks.metadata import C2PA_UUID

        p = self._mp4_late_c2pa(tmp_path)
        assert C2PA_UUID not in p.read_bytes()[: 1024 * 1024]

    def test_scan_head_includes_late_box(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import C2PA_UUID, scan_head

        assert C2PA_UUID in scan_head(self._mp4_late_c2pa(tmp_path))

    def test_has_ai_metadata_detects_late_manifest(self, tmp_path: Path):
        assert has_ai_metadata(self._mp4_late_c2pa(tmp_path)) is True

    def test_scan_c2pa_region_non_isobmff_is_empty(self, tmp_path: Path):
        from remove_ai_watermarks.noai.isobmff import scan_c2pa_region

        p = tmp_path / "not.bin"
        p.write_bytes(b"\x89PNG\r\n\x1a\n not an isobmff file")
        assert scan_c2pa_region(p) == b""

    def test_scan_c2pa_region_reads_largesize_uuid(self, tmp_path: Path):
        """A 64-bit largesize (size32 == 1) uuid box must be walked and collected."""
        import struct

        from remove_ai_watermarks.noai.isobmff import scan_c2pa_region

        payload = b"LARGESIZE-C2PA-MANIFEST"
        total = 16 + len(payload)  # 4 (size32=1) + 4 (type) + 8 (largesize) + payload
        uuid_box = struct.pack(">I", 1) + b"uuid" + struct.pack(">Q", total) + payload
        p = tmp_path / "large.mp4"
        p.write_bytes(_MP4_FTYP + uuid_box)
        assert payload in scan_c2pa_region(p)

    def test_scan_c2pa_region_caps_at_max_total(self, tmp_path: Path):
        """The collected payload is bounded by ``max_total`` (never unbounded)."""
        from remove_ai_watermarks.noai.isobmff import scan_c2pa_region

        p = tmp_path / "big.mp4"
        p.write_bytes(_MP4_FTYP + _box(b"uuid", b"A" * 5000))
        assert len(scan_c2pa_region(p, max_total=1000)) <= 1000

    def test_front_placed_manifest_still_detected(self, tmp_path: Path):
        # Regression: a faststart MP4 (manifest before mdat) is unaffected.
        from remove_ai_watermarks.metadata import C2PA_UUID

        manifest = C2PA_UUID + b"OpenAI ... trainedAlgorithmicMedia ..."
        p = tmp_path / "front.mp4"
        p.write_bytes(_MP4_FTYP + _box(b"uuid", manifest) + _box(b"mdat", b"\x00" * 100))
        assert has_ai_metadata(p) is True


_AI_XMP = (
    b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b'<x:xmpmeta><TC260:AIGC>{"Label":"1"}</TC260:AIGC></x:xmpmeta>'
    b'<?xpacket end="w"?>'
)
_PLAIN_XMP = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b"<x:xmpmeta><dc:rights>(c) me</dc:rights></x:xmpmeta>"
    b'<?xpacket end="w"?>'
)


class TestMetaBoxXmpBlanking:
    """HEIF/AVIF store XMP as a meta-box ``mime`` item (bytes in mdat/idat), out of
    reach of the top-level box stripper. An AI-label XMP packet there is blanked
    in place (same length -> iloc offsets and image data stay intact)."""

    def test_blanks_ai_packet_only(self):
        from remove_ai_watermarks.noai.isobmff import blank_ai_xmp_packets

        before, after = b"IMG_BEFORE" * 4, b"IMG_AFTER" * 4
        data = before + _AI_XMP + after + _PLAIN_XMP
        out, n = blank_ai_xmp_packets(data)
        assert n == 1
        assert len(out) == len(data)  # same length -> no offset shifts
        assert b"TC260:AIGC" not in out  # AI label destroyed
        assert before in out  # surrounding (image) bytes intact
        assert after in out
        assert b"dc:rights" in out  # plain XMP left alone

    def test_no_packet_is_noop(self):
        from remove_ai_watermarks.noai.isobmff import blank_ai_xmp_packets

        data = b"just some mdat bytes, no xmp here"
        assert blank_ai_xmp_packets(data) == (data, 0)

    def test_plain_xmp_untouched(self):
        from remove_ai_watermarks.noai.isobmff import blank_ai_xmp_packets

        out, n = blank_ai_xmp_packets(_PLAIN_XMP)
        assert n == 0
        assert out == _PLAIN_XMP

    def test_remove_ai_metadata_blanks_meta_box_xmp(self, tmp_path: Path):
        # End-to-end: a HEIF with an AI XMP packet inside mdat is cleaned without
        # touching the surrounding (coded image) bytes or the file length.
        heic_ftyp = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"
        img = b"CODEDIMAGE" * 8
        mdat = _box(b"mdat", img + _AI_XMP + img)
        src = tmp_path / "ai.heic"
        src.write_bytes(heic_ftyp + mdat)
        assert has_ai_metadata(src) is True

        out = tmp_path / "clean.heic"
        remove_ai_metadata(src, out)
        res = out.read_bytes()
        assert len(res) == src.stat().st_size  # length preserved
        assert b"TC260:AIGC" not in res
        assert img in res  # coded image bytes intact
        assert has_ai_metadata(out) is False


class TestIsobmffMetadataRemoval:
    """Container-level AI-provenance stripping across ISOBMFF image/video/audio."""

    def test_strips_ai_xmp_uuid_box(self):
        # A uuid box carrying a TC260 AIGC label is dropped by content match,
        # regardless of the (non-C2PA) XMP UUID's byte order.
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        xmp_uuid = bytes(range(16))  # arbitrary, not the C2PA UUID
        payload = b'<x:xmpmeta><TC260:AIGC>{"Label":"1"}</TC260:AIGC></x:xmpmeta>'
        box = (24 + len(payload)).to_bytes(4, "big") + b"uuid" + xmp_uuid + payload
        cleaned, stripped = strip_c2pa_boxes(_MP4_FTYP + box + _MP4_MDAT)
        assert stripped == 1
        assert cleaned == _MP4_FTYP + _MP4_MDAT

    def test_keeps_plain_non_ai_xmp(self):
        # A uuid box with ordinary (non-AI) XMP must be preserved.
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        xmp_uuid = bytes(range(16))
        payload = b"<x:xmpmeta><dc:rights>(c) me</dc:rights></x:xmpmeta>"
        box = (24 + len(payload)).to_bytes(4, "big") + b"uuid" + xmp_uuid + payload
        cleaned, stripped = strip_c2pa_boxes(_MP4_FTYP + box + _MP4_MDAT)
        assert stripped == 0
        assert cleaned == _MP4_FTYP + box + _MP4_MDAT

    def test_m4a_c2pa_stripped(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import C2PA_UUID

        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"manifest"
        src = tmp_path / "voice.m4a"
        src.write_bytes(_MP4_FTYP + uuid_box + _MP4_MDAT)
        out = tmp_path / "clean.m4a"
        remove_ai_metadata(src, out)
        assert out.read_bytes() == _MP4_FTYP + _MP4_MDAT

    def test_content_sniff_routes_unknown_suffix(self, tmp_path: Path):
        # An ISOBMFF file with a non-standard extension is still box-stripped.
        from remove_ai_watermarks.metadata import C2PA_UUID

        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"manifest"
        src = tmp_path / "mystery.bin"
        src.write_bytes(_MP4_FTYP + uuid_box + _MP4_MDAT)
        out = tmp_path / "out.bin"
        remove_ai_metadata(src, out)
        assert out.read_bytes() == _MP4_FTYP + _MP4_MDAT

    def test_unparseable_audio_raises(self, tmp_path: Path):
        # Garbage that ffmpeg can't parse must raise a clear error, not crash in
        # the image path. (When ffmpeg is absent this still raises RuntimeError.)
        src = tmp_path / "audio.mp3"
        src.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00 not real mp3 frames")
        out = tmp_path / "out.mp3"
        with pytest.raises(RuntimeError):
            remove_ai_metadata(src, out)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestFfmpegMetadataStrip:
    """Lossless container-metadata strip for non-ISOBMFF audio/video via ffmpeg."""

    def _wav_with_tag(self, path: Path, tag: str = "Suno AI") -> None:
        subprocess.run(  # noqa: S603
            [
                shutil.which("ffmpeg"),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.1",
                "-metadata",
                f"title={tag}",
                str(path),
            ],
            check=True,
        )

    def test_strips_wav_title_metadata(self, tmp_path: Path):
        src = tmp_path / "in.wav"
        self._wav_with_tag(src, "Suno AI generated")
        assert b"Suno AI generated" in src.read_bytes()  # tag is present pre-strip
        out = tmp_path / "clean.wav"
        remove_ai_metadata(src, out)
        assert out.exists()
        assert b"Suno AI generated" not in out.read_bytes()  # tag stripped, audio kept


class TestC2paCloudManifest:
    """C2PA 2.4 Durable Content Credentials: an XMP dcterms:provenance pointer to
    a vendor cloud manifest store survives when the embedded manifest is stripped."""

    def _cloud_png(self, tmp_path: Path, host: bytes = b"cai-manifests.adobe.com") -> Path:
        xmp = (
            b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description rdf:about="" xmlns:dcterms="http://purl.org/dc/terms/" '
            b'dcterms:provenance="https://' + host + b'/manifests/urn-c2pa-abc123"> </rdf:Description>'
            b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
        )
        p = tmp_path / "cloud.png"
        img = Image.new("RGB", (16, 16))
        meta = PngInfo()
        meta.add_itxt("XML:com.adobe.xmp", xmp.decode("latin-1"))
        img.save(p, pnginfo=meta)
        return p

    def test_detects_adobe_cloud_manifest(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import c2pa_cloud_manifest

        assert c2pa_cloud_manifest(self._cloud_png(tmp_path)) == "Adobe Content Authenticity"

    def test_no_provenance_pointer_is_none(self, tmp_clean_png: Path):
        from remove_ai_watermarks.metadata import c2pa_cloud_manifest

        assert c2pa_cloud_manifest(tmp_clean_png) is None

    def test_unknown_host_is_none(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import c2pa_cloud_manifest

        # A dcterms:provenance pointer to an unrecognized host is not attributed.
        assert c2pa_cloud_manifest(self._cloud_png(tmp_path, host=b"manifests.example.com")) is None

    def test_cloud_manifest_does_not_assert_ai(self, tmp_path: Path):
        # Provenance only -- a cloud manifest can describe a human edit, so the
        # verdict must stay 'unknown', not 'AI-generated'.
        from remove_ai_watermarks.identify import identify

        r = identify(self._cloud_png(tmp_path), check_visible=False, check_invisible=False)
        assert r.is_ai_generated is None
        assert any("Durable Content Credentials" in w for w in r.watermarks)
        assert any(s.name == "c2pa_cloud" for s in r.signals)


def test_remove_all_bypasses_lossless_jpeg(tmp_path, monkeypatch):
    # keep_standard=False (--remove-all) must NOT take the AI-only lossless JPEG path,
    # which leaves standard (non-AI) metadata in place; it must fall through to the full
    # re-encode strip (#7).
    import remove_ai_watermarks.metadata as md

    calls: list[int] = []
    real = md._strip_jpeg_metadata_lossless

    def spy(s, o):
        calls.append(1)
        return real(s, o)

    monkeypatch.setattr(md, "_strip_jpeg_metadata_lossless", spy)
    src = tmp_path / "x.jpg"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(src, "JPEG")
    out = tmp_path / "o.jpg"

    md.remove_ai_metadata(src, out, keep_standard=True)
    assert calls, "lossless JPEG strip should run when keeping standard metadata"
    calls.clear()
    md.remove_ai_metadata(src, out, keep_standard=False)
    assert not calls, "keep_standard=False must bypass the AI-only lossless JPEG path"

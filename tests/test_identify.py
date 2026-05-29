"""Tests for the provenance identifier (identify.py).

Pure attribution logic is unit-tested directly; end-to-end verdicts assert
against the real committed C2PA / IPTC fixtures in data/samples/.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from remove_ai_watermarks.identify import (
    ProvenanceReport,
    _ai_tools_in,
    _attribute_platform,
    _integrity_clashes,
    _issuers_in,
    _vendor_of,
    identify,
)

# Where the lazy import inside identify._visible_sparkle resolves the detector.
_SPARKLE_TARGET = "remove_ai_watermarks.gemini_engine.detect_sparkle_confidence"

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# ── Pure attribution logic (no file IO) ─────────────────────────────


class TestAttributePlatform:
    def test_openai(self):
        assert "OpenAI" in (_attribute_platform(["OpenAI"]) or "")

    def test_designer_wins_over_openai_backend(self):
        # Microsoft Designer signs as "OpenAI, Microsoft"; name the product.
        platform = _attribute_platform(["OpenAI", "Microsoft"])
        assert platform
        assert "Designer" in platform

    def test_adobe(self):
        assert _attribute_platform(["Adobe"]) == "Adobe Firefly"

    def test_google(self):
        assert "Google" in (_attribute_platform(["Google LLC"]) or "")

    def test_truepic_is_signer_not_generator(self):
        platform = _attribute_platform(["Truepic"])
        assert platform
        assert "signer" in platform.lower()

    def test_microsoft_label_is_model_neutral(self):
        # Bing now runs MAI-Image, not DALL-E; the label must not claim DALL-E.
        platform = _attribute_platform(["Microsoft"])
        assert platform
        assert "DALL-E" not in platform

    def test_stability(self):
        platform = _attribute_platform(["Stability AI"])
        assert platform
        assert "Stability AI" in platform

    def test_empty_is_none(self):
        assert _attribute_platform([]) is None


class TestIssuersIn:
    def test_finds_openai(self):
        assert _issuers_in(b"...OpenAI...trainedAlgorithmicMedia") == ["OpenAI"]

    def test_finds_multiple_sorted(self):
        assert _issuers_in(b"Microsoft and OpenAI") == ["Microsoft", "OpenAI"]

    def test_none_present(self):
        assert _issuers_in(b"just some bytes") == []


class TestAiToolsIn:
    def test_finds_generator(self):
        assert _ai_tools_in(b"...claim_generator Imagen 3...") == ["Imagen"]

    def test_none_present(self):
        assert _ai_tools_in(b"a regular photo, no tools") == []


class TestIdentifyNonPng:
    """Non-PNG containers (JPEG/WebP/AVIF) carry C2PA where the caBX parser can't
    reach; identify recovers issuer + generator via the binary scan. Synthetic
    byte blobs mirror tests/test_metadata.py::TestSynthIDSourceNonPng.
    """

    def _c2pa_jpeg(self, tmp_path: Path, blob: bytes) -> Path:
        path = tmp_path / "img.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_google_imagen_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Google Imagen ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Google" in r.platform
        # Generator recovered from the non-PNG blob shows up in the c2pa signal.
        c2pa_signal = next(s for s in r.signals if s.name == "c2pa")
        assert "Imagen" in c2pa_signal.detail

    def test_openai_jpeg_has_synthid(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI DALL-E ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert any("SynthID" in w for w in r.watermarks)

    def test_black_forest_labs_flux_attributed(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Black Forest Labs API ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Black Forest Labs (FLUX)"

    def test_bytedance_volcengine_attributed(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"certificate_center@volcengine.com ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert "ByteDance" in (r.platform or "")

    def test_stability_ai_issuer_attributed_no_synthid(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Stability AI ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Stability AI" in r.platform
        assert not any("SynthID" in w for w in r.watermarks)  # Stability does not use SynthID

    def test_c2pa_without_ai_marker_is_unknown(self, tmp_path: Path):
        # Adobe signs C2PA on plain Photoshop edits too. Without an AI digital-
        # source marker, the honest verdict is unknown -- the C2PA watermark is
        # still listed, but is_ai_generated is not asserted True.
        path = self._c2pa_jpeg(tmp_path, b"Adobe ... no ai marker here")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is None
        assert any("C2PA" in w for w in r.watermarks)
        assert not any("SynthID" in w for w in r.watermarks)


class TestIdentifySamsungGalaxy:
    """Samsung Galaxy / ASUS Gallery C2PA signers (verified on real signed files
    2026-05-29; synthetic byte blobs here since the originals are private).

    Galaxy AI edits stamp BOTH the device cert AND an AI source-type / genAIType,
    so the signer attribution must NOT trip the camera-vs-AI integrity clash.
    """

    def _jpeg(self, tmp_path: Path, name: str, blob: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_galaxy_trained_source_is_high_ai(self, tmp_path: Path):
        path = self._jpeg(tmp_path, "s25.jpg", b"Samsung Galaxy Galaxy S25 c2pa-rs trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform == "Samsung Galaxy (C2PA)"
        assert r.integrity_clashes == []  # device cert + AI source-type is legitimate, not a clash

    def test_galaxy_genai_only_is_medium_ai(self, tmp_path: Path):
        # The Galaxy S24 case: no trainedAlgorithmicMedia, genAIType is the only
        # AI marker -- previously missed, now a medium-confidence verdict.
        path = self._jpeg(
            tmp_path, "s24.jpg", b'Samsung Galaxy Galaxy S24 c2pa-rs PhotoEditor_Re_Edit_Data{"genAIType":1}'
        )
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform == "Samsung Galaxy (C2PA)"
        assert any(s.name == "samsung_genai" for s in r.signals)
        assert r.integrity_clashes == []

    def test_asus_gallery_signer_not_ai(self, tmp_path: Path):
        # ASUS Gallery signs edited photos; no AI source-type or genAIType, so the
        # platform is attributed but the verdict stays unknown.
        path = self._jpeg(tmp_path, "asus.jpg", b"/com.asus.gallery/3.8.0.98 c2pa-rs no ai marker")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is None
        assert r.platform == "ASUS Gallery (C2PA signer)"
        assert any("C2PA" in w for w in r.watermarks)


# ── End-to-end verdicts on real fixtures ────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIdentifyRealSamples:
    def test_openai_chatgpt(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform
        assert "OpenAI" in r.platform
        assert any("C2PA" in w for w in r.watermarks)
        assert any("SynthID" in w for w in r.watermarks)

    def test_adobe_firefly_has_no_synthid(self):
        r = identify(SAMPLES_DIR / "firefly-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Adobe Firefly"
        assert not any("SynthID" in w for w in r.watermarks)

    def test_iptc_made_with_ai(self):
        # mj-1.png carries the IPTC digitalSourceType "Made with AI" marker.
        r = identify(SAMPLES_DIR / "mj-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert any("IPTC" in w for w in r.watermarks)

    def test_clean_photo_is_unknown_not_clean(self, clean_photo: Path):
        r = identify(clean_photo, check_visible=False)
        assert r.is_ai_generated is None  # never asserted False
        assert r.platform is None
        assert r.confidence == "none"
        assert r.watermarks == []

    def test_strip_caveat_always_present(self, clean_photo: Path):
        r = identify(clean_photo, check_visible=False)
        assert any("not proof" in c for c in r.caveats)

    def test_returns_report_dataclass(self):
        assert isinstance(identify(SAMPLES_DIR / "firefly-1.png", check_visible=False), ProvenanceReport)


# ── Local diffusion parameters (Stable Diffusion / ComfyUI) ─────────


class TestIdentifyLocalParams:
    """A PNG carrying SD-style generation params is attributed to a local pipeline."""

    def test_sd_params_attributed_to_local_pipeline(self, tmp_png_with_ai_metadata: Path):
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform is not None
        assert "Stable Diffusion" in r.platform
        assert any("generation parameters" in w for w in r.watermarks)

    def test_gen_params_signal_lists_keys(self, tmp_png_with_ai_metadata: Path):
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        signal = next(s for s in r.signals if s.name == "gen_params")
        assert "parameters" in signal.detail
        assert signal.confidence == "high"

    def test_clean_png_is_unknown(self, tmp_clean_png: Path):
        r = identify(tmp_clean_png, check_visible=False)
        assert r.is_ai_generated is None
        assert r.platform is None
        assert r.confidence == "none"
        assert r.signals == []


# ── China TC260 AIGC label as a PNG text chunk (Doubao) ─────────────


class TestIdentifyAigcPngChunk:
    """The raw-JSON ``AIGC`` PNG chunk (no namespaced XMP marker) is a high-
    confidence AI verdict, same as the XMP form."""

    def _aigc_chunk_png(self, tmp_path: Path) -> Path:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        p = tmp_path / "doubao_chunk.png"
        pnginfo = PngInfo()
        pnginfo.add_text("AIGC", json.dumps({"Label": "1", "ContentProducer": "doubao"}))
        Image.new("RGB", (32, 32)).save(p, pnginfo=pnginfo)
        return p

    def test_png_chunk_detected_high(self, tmp_path: Path):
        r = identify(self._aigc_chunk_png(tmp_path), check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform is not None
        assert "AIGC" in r.platform
        signal = next(s for s in r.signals if s.name == "aigc")
        assert "doubao" in signal.detail


# ── HuggingFace-hosted job marker (medium confidence) ───────────────


class TestIdentifyHuggingFaceJob:
    """The hf-job-id chunk lifts an otherwise-Unknown verdict to a tentative
    (medium) AI, never overriding a high-confidence metadata signal."""

    def _hf_png(self, tmp_path: Path) -> Path:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        p = tmp_path / "hfjob.png"
        pnginfo = PngInfo()
        pnginfo.add_text("hf-job-id", "ec8380a6-2091-423a-b835-209420f99ee1")
        Image.new("RGB", (32, 32)).save(p, pnginfo=pnginfo)
        return p

    def test_hf_job_promotes_to_medium(self, tmp_path: Path):
        r = identify(self._hf_png(tmp_path), check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform is not None
        assert "HuggingFace" in r.platform
        signal = next(s for s in r.signals if s.name == "hf_job")
        assert signal.confidence == "medium"

    def test_hf_job_caveat_present(self, tmp_path: Path):
        r = identify(self._hf_png(tmp_path), check_visible=False)
        assert any("hf-job-id" in c for c in r.caveats)

    def test_metadata_keeps_high_even_with_hf_job(self, tmp_png_with_ai_metadata: Path):
        # A high-confidence metadata verdict is not downgraded by an hf-job hit.
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(tmp_png_with_ai_metadata)
        pnginfo = PngInfo()
        for k, v in img.text.items():
            pnginfo.add_text(k, v)
        pnginfo.add_text("hf-job-id", "ec8380a6-2091-423a-b835-209420f99ee1")
        img.save(tmp_png_with_ai_metadata, pnginfo=pnginfo)
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        assert r.confidence == "high"


# ── Visible-sparkle fallback (mocked detector) ──────────────────────


class TestIdentifyVisibleSparkle:
    """The visible-sparkle signal gates on the corpus-tuned threshold (0.5)."""

    def test_above_threshold_promotes_to_medium(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=0.7):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform is not None
        assert "Gemini" in r.platform
        signal = next(s for s in r.signals if s.name == "visible_sparkle")
        assert signal.confidence == "medium"

    def test_below_threshold_not_promoted(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=0.4):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is None
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_detector_unavailable_does_not_crash(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=None):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is None
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_check_visible_false_skips_detector(self, tmp_clean_png: Path):
        # Even a strong detection is ignored when the caller opts out.
        with patch(_SPARKLE_TARGET, return_value=0.99) as mock_detect:
            r = identify(tmp_clean_png, check_visible=False)
        mock_detect.assert_not_called()
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_metadata_keeps_high_even_with_sparkle(self, tmp_png_with_ai_metadata: Path):
        # Metadata verdict (high) is not downgraded by an additional sparkle hit.
        with patch(_SPARKLE_TARGET, return_value=0.7):
            r = identify(tmp_png_with_ai_metadata, check_visible=True)
        assert r.confidence == "high"


# ── Caveats and serialization ───────────────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIdentifyCaveats:
    def test_openai_hedge_caveat_present(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert any("before the rollout" in c for c in r.caveats)

    def test_synthid_proxy_caveat_present(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert any("not locally" in c for c in r.caveats)

    def test_caveats_are_deduplicated(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert len(r.caveats) == len(set(r.caveats))


class TestReportSerializable:
    def test_report_is_json_serializable(self, tmp_png_with_ai_metadata: Path):
        # The CLI --json path relies on asdict + json.dumps(default=str).
        report = identify(tmp_png_with_ai_metadata, check_visible=False)
        dumped = json.dumps(asdict(report), default=str)
        assert "is_ai_generated" in dumped


class TestIdentifyExifGenerator:
    """An AI generator tag in EXIF/XMP (incl. AVIF) drives attribution."""

    def test_avif_firefly_software_attributed(self, tmp_path: Path):
        import piexif
        from PIL import Image

        exif = piexif.dump({"0th": {piexif.ImageIFD.Software: b"Adobe Firefly"}, "Exif": {}, "GPS": {}, "1st": {}})
        path = tmp_path / "firefly.avif"
        Image.new("RGB", (64, 64), (90, 80, 70)).save(path, exif=exif)
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Firefly" in r.platform
        assert any("generator tag" in w for w in r.watermarks)


class TestIdentifyXaiSignature:
    """xAI / Grok's EXIF Signature + UUID-Artist drives an xAI verdict."""

    def test_grok_signature_attributed(self, tmp_path: Path):
        import piexif
        from PIL import Image

        exif = piexif.dump(
            {
                "0th": {
                    piexif.ImageIFD.ImageDescription: b"Signature: " + b"A" * 120,
                    piexif.ImageIFD.Artist: b"12345678-1234-1234-1234-123456789abc",
                },
                "Exif": {},
                "GPS": {},
                "1st": {},
            }
        )
        path = tmp_path / "grok.jpg"
        Image.new("RGB", (64, 64), (70, 80, 90)).save(path, exif=exif)
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "xAI" in r.platform
        assert any("xAI/Grok" in w for w in r.watermarks)


class TestIdentifySoftBinding:
    """A C2PA soft-binding alg names a forensic-watermark vendor in the inventory."""

    def test_soft_binding_vendor_listed(self, tmp_path: Path):
        p = tmp_path / "sb.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe1 c2pa jumb com.digimarc.validate.1 \xff\xd9")
        r = identify(p, check_visible=False, check_invisible=False)
        assert any("Digimarc" in w for w in r.watermarks)
        assert any(s.name == "soft_binding" for s in r.signals)


class TestIdentifyIptcAi:
    """IPTC 2025.1 AISystemUsed drives an AI verdict + platform attribution."""

    def test_iptc_ai_system_attributed(self, tmp_path: Path):
        p = tmp_path / "iptc.jpg"
        p.write_bytes(
            b"\xff\xd8\xff\xe1<x:xmpmeta><Iptc4xmpExt:AISystemUsed>Google Gemini"
            b"</Iptc4xmpExt:AISystemUsed></x:xmpmeta>\xff\xd9"
        )
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Gemini" in r.platform


class TestIdentifyC2paDevice:
    """A distinctive C2PA device token wins platform attribution over incidental
    issuer-name mentions (regression guard for real-sample mis-attribution:
    Leica->Truepic, Nikon->Adobe, Pixel->Google Gemini)."""

    def test_leica_token_beats_incidental_tokens(self, tmp_path: Path):
        # "Adobe"/"Google"/"Truepic" appear incidentally; Leica's lc_c2pa wins.
        blob = b"\xff\xd8\xff\xe1 c2pa.claim jumbf Adobe Google Truepic lc_c2pa \xff\xd9"
        p = tmp_path / "leica_like.jpg"
        p.write_bytes(blob)
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.platform == "Leica (camera, C2PA capture)"

    def test_pixel_camera_cert_beats_incidental_google(self, tmp_path: Path):
        # Pixel's cert CN is "Pixel Camera"; "Google LLC" appears as the cert org
        # but must NOT yield "Google (Gemini / Imagen)" -- it is a camera capture.
        blob = b"\xff\xd8\xff\xe1 c2pa.claim jumbf Google LLC Adobe Pixel Camera \xff\xd9"
        p = tmp_path / "pixel_like.jpg"
        p.write_bytes(blob)
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.platform == "Google Pixel (camera, C2PA capture)"
        assert r.is_ai_generated is None  # camera capture, not AI

    def test_sony_namespace_beats_bare_make(self, tmp_path: Path):
        # Sony's own C2PA assertion namespace (sony.sig), not the bare "Sony"
        # EXIF Make that appears on ordinary photos.
        blob = b"\xff\xd8\xff\xe1 c2pa.claim jumbf Adobe Sony sony.sig.v1_1 \xff\xd9"
        p = tmp_path / "sony_like.jpg"
        p.write_bytes(blob)
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.platform == "Sony (camera, C2PA capture)"

    def test_unmapped_device_not_mislabeled_via_incidental_issuer(self, tmp_path: Path):
        # An unmapped camera (Canon) whose manifest incidentally contains the
        # "Adobe" XMP-toolkit string, with NO AI source type, must NOT be labeled
        # "Adobe Firefly". The issuer->generator mapping only applies to AI content.
        blob = b"\xff\xd8\xff\xe1 c2pa.claim jumbf Canon EOS Adobe XMP Core \xff\xd9"
        p = tmp_path / "canon_like.jpg"
        p.write_bytes(blob)
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is None  # camera capture, not AI
        assert r.platform is not None
        assert "Firefly" not in r.platform  # not mislabeled as an AI generator


# ── Open invisible watermark (SD/SDXL/FLUX) integration ─────────────

from remove_ai_watermarks.invisible_watermark import is_available as _wm_available  # noqa: E402


@pytest.mark.skipif(not _wm_available(), reason="invisible-watermark not installed")
class TestIdentifyInvisibleWatermark:
    def _sdxl_watermarked(self, tmp_path: Path) -> Path:
        import cv2
        import numpy as np
        from imwatermark import WatermarkEncoder

        from remove_ai_watermarks.invisible_watermark import _BITS_48

        bits = [int(b) for b in format(_BITS_48["Stable Diffusion XL"], "048b")]
        enc = WatermarkEncoder()
        enc.set_watermark("bits", bits)
        img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
        path = tmp_path / "sdxl.png"
        cv2.imwrite(str(path), enc.encode(img, "dwtDct"))
        return path

    def test_sdxl_watermark_identified(self, tmp_path: Path):
        r = identify(self._sdxl_watermarked(tmp_path), check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform is not None
        assert "Stable Diffusion XL" in r.platform
        assert any("invisible watermark" in w.lower() for w in r.watermarks)

    def test_check_invisible_false_skips(self, tmp_path: Path):
        r = identify(self._sdxl_watermarked(tmp_path), check_visible=False, check_invisible=False)
        assert not any(s.name == "invisible_watermark" for s in r.signals)


class TestIdentifyAIGC:
    """China TC260 AIGC label is detected and attributed (e.g. Doubao)."""

    def _aigc_png(self, tmp_path: Path) -> Path:
        from PIL import Image

        p = tmp_path / "doubao.png"
        Image.new("RGB", (32, 32)).save(p)
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/">'
            "<TC260:AIGC>{&quot;Label&quot;:&quot;1&quot;,&quot;ContentProducer&quot;:&quot;BYTEDANCE001&quot;}"
            "</TC260:AIGC></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        with open(p, "ab") as f:
            f.write(xmp.encode())
        return p

    def test_aigc_detected(self, tmp_path: Path):
        r = identify(self._aigc_png(tmp_path), check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "AIGC" in r.platform or "TC260" in r.platform
        assert any("AIGC" in w for w in r.watermarks)

    def test_aigc_signal_carries_producer(self, tmp_path: Path):
        r = identify(self._aigc_png(tmp_path), check_visible=False)
        sig = next(s for s in r.signals if s.name == "aigc")
        assert "BYTEDANCE001" in sig.detail


# ── Integrity clashes (contradictions between independent signals) ──────


class TestVendorOf:
    def test_openai_variants(self):
        assert _vendor_of("OpenAI (ChatGPT / gpt-image / DALL-E / Sora)") == "OpenAI"
        assert _vendor_of("DALL-E 3") == "OpenAI"

    def test_google_variants(self):
        assert _vendor_of("Google (Gemini / Imagen)") == "Google"
        assert _vendor_of("Imagen 3") == "Google"

    def test_other_vendors(self):
        assert _vendor_of("Ideogram AI") == "Ideogram"
        assert _vendor_of("Adobe Firefly") == "Adobe"
        assert _vendor_of("Stability AI (Stable Image)") == "Stability AI"

    def test_camera_label_is_not_an_ai_vendor(self):
        # Camera platform labels must NOT normalize to an AI vendor, or a camera
        # capture would be mistaken for AI-generation in clash detection.
        assert _vendor_of("Leica (camera, C2PA capture)") is None

    def test_unknown_is_none(self):
        assert _vendor_of("a regular photo") is None
        assert _vendor_of(None) is None


class TestIntegrityClashesHelper:
    def test_two_ai_vendors_clash(self):
        clashes = _integrity_clashes({"c2pa": "OpenAI", "exif_generator": "Ideogram"}, None, camera_has_ai_marker=True)
        assert len(clashes) == 1
        assert "OpenAI" in clashes[0]
        assert "Ideogram" in clashes[0]

    def test_same_vendor_two_signals_no_clash(self):
        # C2PA Google + SynthID-Google proxy is consistent, not a contradiction.
        assert _integrity_clashes({"c2pa": "Google", "synthid": "Google"}, None, camera_has_ai_marker=True) == []

    def test_single_vendor_no_clash(self):
        assert _integrity_clashes({"c2pa": "OpenAI"}, None, camera_has_ai_marker=True) == []

    def test_empty_no_clash(self):
        assert _integrity_clashes({}, None, camera_has_ai_marker=False) == []

    def test_camera_plus_ai_marker_clashes(self):
        clashes = _integrity_clashes(
            {"exif_generator": "Ideogram"},
            "Google Pixel (camera, C2PA capture)",
            camera_has_ai_marker=True,
        )
        assert any("Camera-capture" in c and "Pixel" in c for c in clashes)

    def test_camera_without_ai_marker_no_clash(self):
        # A clean camera capture (the normal case for our Pixel/Leica/Sony files)
        # must NOT raise a clash.
        assert _integrity_clashes({}, "Leica (camera, C2PA capture)", camera_has_ai_marker=False) == []


class TestIntegrityClashEndToEnd:
    def _c2pa_jpeg(self, tmp_path: Path, blob: bytes) -> Path:
        path = tmp_path / "img.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_two_generator_stamps_clash(self, tmp_path: Path):
        # An OpenAI C2PA manifest (AI source) on an image that ALSO carries a
        # China TC260 AIGC label = two independent generator stamps naming
        # different origins -> a laundering tell.
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia ... TC260:AIGC label")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes
        assert any("Conflicting AI-origin" in c for c in r.integrity_clashes)

    def test_single_stamp_no_clash(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes == []

    def test_clash_serializes_to_json(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia ... TC260:AIGC label")
        r = identify(path, check_visible=False, check_invisible=False)
        payload = json.loads(json.dumps(asdict(r), default=str))
        assert payload["integrity_clashes"] == r.integrity_clashes


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
@pytest.mark.parametrize("fixture", ["chatgpt-1.png", "firefly-1.png", "doubao-1.png", "grok-1.jpg", "mj-1.png"])
class TestRealSamplesHaveNoClash:
    """Every real single-origin fixture must report zero clashes (false-positive guard)."""

    def test_no_false_positive_clash(self, fixture: str):
        path = SAMPLES_DIR / fixture
        if not path.exists():
            pytest.skip(f"{fixture} not present")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes == []

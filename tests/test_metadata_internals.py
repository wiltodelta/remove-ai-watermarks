"""Tests for metadata compatibility submodules: constants, extractor, C2PA, plus the
consolidated metadata strip (formerly legacy metadata helper)."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import pytest
from PIL import Image

from remove_ai_watermarks._internal.c2pa import (
    _parse_c2pa_chunk,
    c2pa_info_from_manifest_store,
    c2pa_info_has_invismark,
    c2pa_info_has_removal_hint,
    cbor_text_after,
    extract_c2pa_chunk,
    extract_c2pa_info,
    has_c2pa_metadata,
    inject_c2pa_chunk,
    synthid_verdict,
)
from remove_ai_watermarks._internal.constants import (
    AI_KEYWORDS,
    AI_METADATA_KEYS,
    C2PA_CHUNK_TYPE,
    PNG_SIGNATURE,
    SUPPORTED_FORMATS,
)
from remove_ai_watermarks._internal.extractor import (
    extract_ai_metadata,
    extract_metadata,
    get_ai_metadata_summary,
    has_ai_metadata,
)
from remove_ai_watermarks._internal.isobmff import (
    blank_ai_exif_tokens,
    is_isobmff,
    strip_c2pa_boxes,
)
from remove_ai_watermarks.metadata import (
    remove_ai_metadata as remove_metadata,
)

# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    """Verify constant integrity."""

    def test_supported_formats_include_png(self):
        assert ".png" in SUPPORTED_FORMATS

    def test_supported_formats_include_jpg(self):
        assert ".jpg" in SUPPORTED_FORMATS

    def test_supported_formats_include_heic_avif(self):
        # HEIC/AVIF are first-class when the visible pixel extra is installed
        # (read+write via pillow-heif), so batch discovers them without a warning.
        assert {".heic", ".heif", ".avif"} <= SUPPORTED_FORMATS

    def test_supported_formats_exclude_jpeg_xl(self):
        # JPEG-XL stays metadata/strip-only -- no pixel decoder without pillow-jxl.
        assert ".jxl" not in SUPPORTED_FORMATS

    def test_ai_metadata_keys_not_empty(self):
        assert len(AI_METADATA_KEYS) > 0

    def test_ai_keywords_not_empty(self):
        assert len(AI_KEYWORDS) > 0

    def test_png_signature_bytes(self):
        assert PNG_SIGNATURE == b"\x89PNG\r\n\x1a\n"

    def test_c2pa_chunk_type(self):
        assert C2PA_CHUNK_TYPE == b"caBX"


# ── Extractor ───────────────────────────────────────────────────────


class TestExtractor:
    """Tests for internal metadata extraction helpers."""

    def test_extract_metadata_returns_dict(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert isinstance(meta, dict)

    def test_extract_metadata_gets_standard_keys(self, tmp_clean_png):
        meta = extract_metadata(tmp_clean_png)
        assert "Author" in meta

    def test_extract_ai_metadata_from_ai_image(self, tmp_png_with_ai_metadata):
        meta = extract_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta

    def test_extract_ai_metadata_from_clean_image(self, tmp_clean_png):
        meta = extract_ai_metadata(tmp_clean_png)
        assert len(meta) == 0

    def test_has_ai_metadata_detects(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_has_ai_metadata_clean(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_summary_with_ai(self, tmp_png_with_ai_metadata):
        summary = get_ai_metadata_summary(tmp_png_with_ai_metadata)
        assert "AI Image Metadata" in summary

    def test_summary_clean(self, tmp_clean_png):
        summary = get_ai_metadata_summary(tmp_clean_png)
        assert "No AI metadata" in summary


# ── Cleaner ─────────────────────────────────────────────────────────


class TestCleaner:
    """Metadata stripping via the single, consolidated ``metadata.remove_ai_metadata``
    (the legacy ``legacy metadata helper`` duplicate was retired)."""

    def test_remove_ai_metadata(self, tmp_png_with_ai_metadata, tmp_path):
        output = tmp_path / "cleaned.png"
        remove_metadata(tmp_png_with_ai_metadata, output)
        assert output.exists()
        # Verify AI metadata removed
        meta = extract_ai_metadata(output)
        assert "parameters" not in meta

    def test_has_ai_content(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)


# ── C2PA ────────────────────────────────────────────────────────────


class TestC2PA:
    """Tests for C2PA detection on regular (non-C2PA) images."""

    def test_no_c2pa_on_regular_png(self, tmp_clean_png):
        assert not has_c2pa_metadata(tmp_clean_png)

    def test_no_c2pa_on_jpeg(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)

    def test_extract_c2pa_none_on_regular(self, tmp_clean_png):
        assert extract_c2pa_chunk(tmp_clean_png) is None

    def test_extract_c2pa_info_empty(self, tmp_clean_png):
        info = extract_c2pa_info(tmp_clean_png)
        assert info == {}

    def test_c2pa_returns_false_for_non_png(self, tmp_jpeg_path):
        assert not has_c2pa_metadata(tmp_jpeg_path)

    def test_structured_extraction_ignores_unreachable_manifests(self):
        store = {
            "active_manifest": "active",
            "manifests": {
                "active": {
                    "signature_info": {"issuer": "Adobe"},
                    "assertions": [],
                },
                "unreachable": {
                    "signature_info": {"issuer": "OpenAI"},
                    "assertions": [
                        {
                            "label": "c2pa.actions.v2",
                            "data": {
                                "actions": [
                                    {
                                        "action": "c2pa.created",
                                        "digitalSourceType": "trainedAlgorithmicMedia",
                                    }
                                ]
                            },
                        }
                    ],
                },
            },
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["issuer"] == "Adobe"
        assert "source_type" not in info
        assert "ai_source_kind" not in info
        assert "c2pa_identity_ai" not in info

    def test_reachable_ingredient_claim_generator_can_assert_ai(self):
        store = {
            "active_manifest": "update",
            "manifests": {
                "update": {
                    "claim_generator": "c2pa-tool/0.1.0",
                    "ingredients": [{"active_manifest": "created"}],
                    "assertions": [],
                },
                "created": {
                    "claim_generator": "Dreamina/7.5.0",
                    "assertions": [],
                },
            },
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["ai_tool"] == "Dreamina"
        assert info["c2pa_identity_ai"] is True

    def test_structured_xai_signer_identity_is_normalized(self):
        store = {
            "active_manifest": "active",
            "manifests": {
                "active": {
                    "signature_info": {
                        "issuer": "Self-signed ephemeral certificate (Content Authenticity SDK) -- LOCAL USE ONLY",
                        "common_name": "xAI Grok Imagine",
                    },
                    "assertions": [],
                }
            },
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["issuer"] == "xAI Grok Imagine"
        assert info["c2pa_identity_ai"] is True

    @pytest.mark.parametrize(
        ("signature", "issuer"),
        [
            ({"issuer": "Bytedance Pte. Ltd."}, "ByteDance"),
            ({"issuer": "TikTok Inc."}, "TikTok"),
        ],
    )
    def test_structured_publisher_identity_does_not_assert_ai(self, signature, issuer):
        store = {
            "active_manifest": "active",
            "manifests": {"active": {"signature_info": signature, "assertions": []}},
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["issuer"] == issuer
        assert "c2pa_identity_ai" not in info

    def test_structured_invismark_exposes_algorithm_and_watermark_id(self):
        watermark_id = "83424621-03cb-40e3-9808-a9fae837156d"
        store = {
            "active_manifest": "paint",
            "manifests": {
                "paint": {
                    "assertions": [
                        {
                            "label": "c2pa.soft-binding",
                            "data": {
                                "alg": "com.microsoft.invismark.1",
                                "blocks": [
                                    {
                                        "scope": "the entire image",
                                        "value": watermark_id,
                                    }
                                ],
                            },
                        }
                    ]
                }
            },
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["soft_binding"] == "Microsoft InvisMark"
        assert info["soft_binding_algorithm"] == "com.microsoft.invismark.1"
        assert info["soft_binding_value"] == watermark_id

    def test_soft_binding_value_requires_its_algorithm(self):
        store = {
            "active_manifest": "broken",
            "manifests": {
                "broken": {
                    "assertions": [
                        {
                            "label": "c2pa.soft-binding",
                            "data": {"blocks": [{"value": "not-attributable"}]},
                        }
                    ]
                }
            },
        }

        assert "soft_binding_value" not in c2pa_info_from_manifest_store(store)

    def test_soft_binding_keeps_invisible_removal_fail_safe(self):
        info = {"soft_binding_vendors": ["Microsoft InvisMark"]}

        assert c2pa_info_has_invismark(info) is True
        assert c2pa_info_has_removal_hint(info) is True

    def test_content_fingerprint_does_not_trigger_invisible_removal(self):
        info = {
            "soft_binding": "Adobe Image Comparator Network",
            "soft_binding_vendors": ["Adobe Image Comparator Network"],
        }

        assert c2pa_info_has_removal_hint(info) is False

    @pytest.mark.parametrize(
        "ingredient_failure",
        [
            # One exclusion rule, reached through two different dimensions: a broken
            # binding and a credential the issuer disowned. The walk classified only the
            # first for a while, so a revoked child manifest stayed reachable and kept
            # donating its claim generator to the parent's attribution.
            "assertion.dataHash.mismatch",
            "signingCredential.ocsp.revoked",
        ],
    )
    def test_invalid_ingredient_does_not_taint_active_validation_or_supply_claims(self, ingredient_failure: str):
        store = {
            "active_manifest": "update",
            "validation_results": {
                "activeManifest": {
                    "success": [
                        {"code": "assertion.dataHash.match"},
                        {"code": "claimSignature.validated"},
                    ],
                    "failure": [{"code": "signingCredential.untrusted"}],
                },
                "ingredientDeltas": [{"validationDeltas": {"failure": [{"code": ingredient_failure}]}}],
            },
            "manifests": {
                "update": {
                    "claim_generator": "c2pa-tool/0.1.0",
                    "ingredients": [
                        {
                            "active_manifest": "created",
                            "validation_results": {"activeManifest": {"failure": [{"code": ingredient_failure}]}},
                        }
                    ],
                    "assertions": [],
                },
                "created": {
                    "claim_generator": "Dreamina/7.5.0",
                    "assertions": [],
                },
            },
        }

        info = c2pa_info_from_manifest_store(store)

        assert info["c2pa_integrity"] == "valid"
        assert info["c2pa_signature"] == "valid"
        assert info["c2pa_signer_trust"] == "untrusted"
        assert "ai_tool" not in info
        assert "c2pa_identity_ai" not in info


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance"
CURRENT_OPENAI_SAMPLE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "synthid"
    / "originals"
    / "ChatGPT Image May 30, 2026, 10_31_08 AM.png"
)


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/fixtures/provenance not present")
class TestC2PARealSamples:
    """Parser behavior on real committed C2PA images."""

    def test_detects_c2pa_in_openai_png(self):
        assert has_c2pa_metadata(SAMPLES_DIR / "chatgpt-1.png")

    def test_extract_info_openai_fields(self):
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        assert info["has_c2pa"] is True
        assert "OpenAI" in info["issuer"]
        assert "c2pa_manifest" in info  # "C2PA manifest (N bytes)"
        assert "trainedAlgorithmicMedia" in info["source_type"]
        # CBOR-clean claim generator, no regex artifacts (e.g. "fGPT-4o").
        assert info["claim_generator"]
        assert not info["claim_generator"].startswith("f")
        assert "synthid_watermark" not in info

    @pytest.mark.skipif(not CURRENT_OPENAI_SAMPLE.exists(), reason="current OpenAI SynthID fixture not present")
    def test_current_openai_watermark_action_asserts_synthid(self):
        info = extract_c2pa_info(CURRENT_OPENAI_SAMPLE)
        assert info["watermarked"] is True
        assert "watermarked.unbound" in info["actions"]
        assert "OpenAI" in info["synthid_watermark"]

    def test_extract_info_adobe_has_no_synthid(self):
        info = extract_c2pa_info(SAMPLES_DIR / "firefly-1.png")
        assert "Adobe" in info["issuer"]
        assert "synthid_watermark" not in info

    def test_extract_chunk_returns_bytes(self):
        chunk = extract_c2pa_chunk(SAMPLES_DIR / "chatgpt-1.png")
        assert chunk is not None
        assert chunk[4:8] == b"caBX"  # chunk type in the 8-byte header

    def test_inject_round_trip(self, tmp_clean_png, tmp_path):
        """Extract a real C2PA chunk, inject into a clean PNG, re-detect."""
        chunk = extract_c2pa_chunk(SAMPLES_DIR / "chatgpt-1.png")
        out = tmp_path / "injected.png"
        inject_c2pa_chunk(tmp_clean_png, out, chunk)
        assert has_c2pa_metadata(out)
        assert "OpenAI" in extract_c2pa_info(out)["issuer"]

    def test_extract_info_flux_jpeg_via_reader(self):
        """Real committed JPEG-with-C2PA fixture: the non-PNG reader path works."""
        info = extract_c2pa_info(SAMPLES_DIR / "flux-1.jpg")
        assert info["has_c2pa"] is True
        assert info["c2pa_manifest"].startswith("C2PA manifest store")  # reader, not chunk
        assert "Black Forest Labs" in info["issuer"]
        assert "trainedAlgorithmicMedia" in info["source_type"]

    def test_extract_info_uses_reader_store(self):
        """The c2pa-python reader path: structured (not heuristic) extraction."""
        from remove_ai_watermarks._internal import c2pa

        assert c2pa.reader_available()
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        # The store-JSON label proves the reader path served this, not the
        # caBX-chunk fallback ("C2PA manifest (...)").
        assert info["c2pa_manifest"].startswith("C2PA manifest store")
        # Structured claim generator is exact, not a CBOR-scanned best-effort.
        assert info["claim_generator"] == "ChatGPT"

    def test_reader_reports_intact_but_untrusted_credentials(self):
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")

        assert info["c2pa_integrity"] == "valid"
        assert info["c2pa_signature"] == "valid"
        assert info["c2pa_signer_trust"] == "untrusted"
        assert info["c2pa_signer_validity"] == "expired"
        assert "assertion.dataHash.match" in info["c2pa_validation_codes"]

    def test_reader_reports_post_signing_container_mutation(self, tampered_chatgpt_png):
        info = extract_c2pa_info(tampered_chatgpt_png)

        assert info["c2pa_integrity"] == "invalid"
        assert info["c2pa_signature"] == "valid"
        assert "assertion.dataHash.mismatch" in info["c2pa_validation_codes"]

    def test_fallback_to_png_parser_when_reader_unavailable(self, monkeypatch):
        """With the reader disabled, the hand-rolled PNG parser still works."""
        from remove_ai_watermarks._internal import c2pa

        monkeypatch.setattr(c2pa, "_C2PA_READER_AVAILABLE", False)
        info = extract_c2pa_info(SAMPLES_DIR / "chatgpt-1.png")
        assert info["c2pa_manifest"].startswith("C2PA manifest (")  # chunk path
        assert "OpenAI" in info["issuer"]
        assert "trainedAlgorithmicMedia" in info["source_type"]
        assert "synthid_watermark" not in info
        assert info["c2pa_integrity"] == "unknown"
        assert info["c2pa_validation_source"] == "fallback"


class TestC2PAInjectValidation:
    def test_inject_rejects_non_png(self, tmp_path):
        with pytest.raises(ValueError, match="only supported for PNG"):
            inject_c2pa_chunk(tmp_path / "in.jpg", tmp_path / "out.png", b"")


# ── CBOR text extraction (parser internals) ─────────────────────────


class TestCborTextAfter:
    """cbor_text_after handles the three CBOR text-string length prefixes."""

    def test_direct_length(self):
        # major-type 3, direct length (0x60 + len). "abc" -> 0x63.
        payload = b"name" + bytes([0x63]) + b"abc"
        assert cbor_text_after(payload, b"name") == "abc"

    def test_one_byte_length(self):
        s = b"x" * 30
        payload = b"name" + bytes([0x78, 30]) + s
        assert cbor_text_after(payload, b"name") == "x" * 30

    def test_two_byte_length(self):
        s = b"y" * 300
        payload = b"name" + bytes([0x79]) + struct.pack(">H", 300) + s
        assert cbor_text_after(payload, b"name") == "y" * 300

    def test_key_not_found_returns_none(self):
        assert cbor_text_after(b"nothing here", b"name") is None

    def test_key_at_end_returns_none(self):
        assert cbor_text_after(b"prefixname", b"name") is None

    def test_invalid_head_returns_none(self):
        # 0x00 is not a text-string head.
        assert cbor_text_after(b"name" + bytes([0x00]) + b"abc", b"name") is None

    def test_latin1_fallback_on_invalid_utf8(self):
        payload = b"name" + bytes([0x61]) + b"\xff"  # len 1, invalid utf-8
        assert cbor_text_after(payload, b"name") is not None


class TestSynthIDVerdict:
    def test_format(self):
        assert synthid_verdict("OpenAI") == "present according to OpenAI provenance"

    def test_multiple_vendors(self):
        assert "Google LLC, OpenAI" in synthid_verdict("Google LLC, OpenAI")


class TestParseChunkGuards:
    """_parse_c2pa_chunk rejects non-printable claim_generator garbage.

    On some manifests (observed: Microsoft Designer) the first ``name`` key
    precedes a binary hash field, not the generator string. The clean issuer +
    SynthID verdict must still come through.
    """

    def test_clean_generator_kept(self):
        # "name" + CBOR text-string (head 0x69 = 0x60+9) "gpt-image"
        chunk = b"...name" + bytes([0x69]) + b"gpt-image" + b"OpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert info["claim_generator"] == "gpt-image"
        assert "OpenAI" in info["issuer"]
        assert "synthid_watermark" in info  # OpenAI + trainedAlgorithmicMedia

    def test_nonprintable_generator_dropped(self):
        # "name" + CBOR string (head 0x64 = len 4) with a control byte -> garbage
        chunk = b"...name" + bytes([0x64]) + b"\x81abc" + b"OpenAI trainedAlgorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert "claim_generator" not in info  # control-char garbage rejected
        assert "OpenAI" in info["issuer"]  # issuer byte-search still robust


class TestC2PADigitalSourceType:
    """The three IPTC digitalSourceType variants drive the AI verdict.

    Only *trained* and *composite-with-trained* mean AI-generated (and so imply
    SynthID provenance for a supported vendor); plain ``algorithmicMedia`` is
    procedural (not trained) and must NOT be flagged as AI.
    """

    def test_plain_algorithmic_media_not_flagged_ai(self):
        chunk = b"...name" + bytes([0x69]) + b"some-tool" + b" OpenAI algorithmicMedia"
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert info["source_type"] == "algorithmicMedia"
        assert "synthid_watermark" not in info  # procedural, not AI-generated

    def test_composite_with_trained_is_ai_and_synthid(self):
        chunk = (
            b"...name"
            + bytes([0x69])
            + b"some-tool"
            + b" OpenAI compositeWithTrainedAlgorithmicMedia c2pa.watermarked.unbound"
        )
        info: dict = {}
        _parse_c2pa_chunk(chunk, info)
        assert "compositeWithTrainedAlgorithmicMedia" in info["source_type"]
        assert "synthid_watermark" in info  # AI-enhanced + OpenAI issuer

    def test_composite_and_bare_algorithmic_cooccur_is_ai(self):
        """Regression: a manifest carrying BOTH ``compositeWithTrainedAlgorithmicMedia``
        (AI-enhanced) and a bare procedural ``algorithmicMedia`` token must classify as
        AI-enhanced. Before the reorder the bare-token elif fired first and returned
        non-AI, dropping the composite AI signal (a false negative)."""
        from remove_ai_watermarks._internal.c2pa import _populate_registry_fields

        info: dict = {}
        _populate_registry_fields(b"x compositeWithTrainedAlgorithmicMedia x algorithmicMedia x", info)
        assert info.get("ai_source_kind") == "enhanced"
        assert "compositeWithTrainedAlgorithmicMedia" in info["source_type"]


# ── ISOBMFF (AVIF / HEIF / JPEG-XL container stripping) ──────────────

FTYP = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"  # 24-byte ftyp box


class TestISOBMFF:
    def test_is_isobmff_true(self):
        assert is_isobmff(FTYP)

    def test_is_isobmff_false_for_png(self):
        assert not is_isobmff(b"\x89PNG\r\n\x1a\n\x00\x00")

    def test_is_isobmff_false_for_short(self):
        assert not is_isobmff(b"abc")

    def test_strips_jpegxl_jumb_box(self):
        """JPEG-XL stores JUMBF in a ``jumb`` box, always stripped."""
        jumb = struct.pack(">I", 8 + 5) + b"jumb" + b"hello"
        cleaned, stripped = strip_c2pa_boxes(FTYP + jumb)
        assert stripped == 1
        assert cleaned == FTYP

    def test_keeps_non_c2pa_box_with_64bit_size(self):
        """size==1 means a 64-bit largesize follows; non-C2PA box is kept."""
        payload = b"\x00" * 8
        box = b"\x00\x00\x00\x01" + b"free" + struct.pack(">Q", 16 + len(payload)) + payload
        cleaned, stripped = strip_c2pa_boxes(FTYP + box)
        assert stripped == 0
        assert cleaned == FTYP + box

    def test_malformed_box_does_not_crash(self):
        # A box claiming size 4 (< 8-byte header) must terminate iteration safely.
        cleaned, stripped = strip_c2pa_boxes(FTYP + b"\x00\x00\x00\x04XXXX")
        assert stripped == 0
        assert cleaned.startswith(FTYP)

    def test_size_zero_box_runs_to_eof(self):
        # size32==0 means the box extends to EOF; a non-C2PA box round-trips.
        box = struct.pack(">I", 0) + b"free" + b"\x00\x00\x00\x00"
        cleaned, stripped = strip_c2pa_boxes(FTYP + box)
        assert stripped == 0
        assert cleaned == FTYP + box

    def test_truncated_largesize_terminates_safely(self):
        # size32==1 promises a 64-bit largesize, but the box ends after 8 bytes;
        # iteration must stop rather than read the missing largesize past EOF.
        # The walk halts before EOF, so the fail-safe returns the input unchanged
        # (emitting only FTYP would silently truncate the file).
        data = FTYP + b"\x00\x00\x00\x01uuid"
        cleaned, stripped = strip_c2pa_boxes(data)
        assert stripped == 0
        assert cleaned == data

    @staticmethod
    def _avif_with_exif(exif_0th: dict) -> bytes:
        """A fake AVIF (ftyp + mdat) whose mdat carries an EXIF TIFF block, as a
        HEIF/AVIF ``Exif`` meta-box item stores it (bytes in mdat)."""
        import piexif

        blob = piexif.dump({"0th": exif_0th})
        mdat = struct.pack(">I", 8 + len(blob)) + b"mdat" + blob
        return FTYP + mdat

    def test_blank_ai_token_in_exif_item(self):
        import piexif

        data = self._avif_with_exif({piexif.ImageIFD.Software: b"DALL-E", piexif.ImageIFD.Make: b"Canon"})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked == 1
        assert len(out) == len(data)  # same length -> box sizes / iloc stay valid
        assert b"DALL-E" not in out  # AI token destroyed
        assert b"Canon" in out  # camera tag preserved
        # The TIFF structure still parses, with the AI value blanked and Make kept.
        blob = out[out.index(b"Exif\x00\x00") + 6 :]
        ifd = piexif.load(blob)["0th"]
        assert ifd[piexif.ImageIFD.Software].strip() == b""
        assert ifd[piexif.ImageIFD.Make] == b"Canon"

    def test_blank_aigc_block_in_exif(self):
        """Parity with the JPEG path: the China TC260 ``{"AIGC":{...}}`` block in EXIF
        ImageDescription must be blanked on the ISOBMFF path too -- ``blank_ai_exif_tokens``
        is the ONLY EXIF scrubber for HEIC/AVIF (``_scrub_ai_exif`` never runs there)."""
        import piexif

        aigc = b'{"AIGC":{"Label":"1","ContentProducer":"00119144030008867405X210002","ProduceID":"abc"}}'
        data = self._avif_with_exif({piexif.ImageIFD.ImageDescription: aigc, piexif.ImageIFD.Make: b"Canon"})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked >= 1
        assert len(out) == len(data)  # same length -> box sizes / iloc stay valid
        assert b'"AIGC"' not in out  # TC260 block destroyed
        assert b"Canon" in out  # camera tag preserved

    def test_blank_xai_signature_pair_in_exif(self):
        """Parity: the xAI/Grok ``Signature:`` blob + UUID ``Artist`` pair in EXIF is
        dropped together on the ISOBMFF path too."""
        import piexif

        sig = b"Signature: " + b"A" * 80
        art = b"12345678-1234-1234-1234-123456789012"
        data = self._avif_with_exif({piexif.ImageIFD.ImageDescription: sig, piexif.ImageIFD.Artist: art})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked == 2  # both the signature and the UUID artist
        assert len(out) == len(data)
        assert b"Signature: AAAA" not in out

    def test_blank_leaves_clean_exif_untouched(self):
        import piexif

        data = self._avif_with_exif({piexif.ImageIFD.Software: b"Adobe Photoshop", piexif.ImageIFD.Make: b"NIKON"})
        out, blanked = blank_ai_exif_tokens(data)
        assert blanked == 0
        assert out == data  # no AI token -> byte-for-byte unchanged

    def test_blank_no_exif_is_noop(self):
        out, blanked = blank_ai_exif_tokens(FTYP + b"\x00\x00\x00\x0cmdat" + b"pixels!!")
        assert blanked == 0
        assert out == FTYP + b"\x00\x00\x00\x0cmdat" + b"pixels!!"

    def test_streaming_malformed_walk_copies_input_unchanged(self, tmp_path: Path):
        from remove_ai_watermarks._internal.isobmff import strip_isobmff_media_file

        source = tmp_path / "malformed.mp4"
        output = tmp_path / "clean.mp4"
        malformed = FTYP + struct.pack(">I", 999) + b"uuid" + b"short"
        source.write_bytes(malformed)

        stripped, tc260_blanked = strip_isobmff_media_file(source, output)

        assert (stripped, tc260_blanked) == (0, 0)
        assert output.read_bytes() == malformed

    def test_streaming_failure_does_not_publish_partial_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from remove_ai_watermarks import metadata
        from remove_ai_watermarks._internal import isobmff

        source = tmp_path / "source.mp4"
        output = tmp_path / "clean.mp4"
        uuid_box = struct.pack(">I", 24) + b"uuid" + metadata.C2PA_UUID
        source.write_bytes(FTYP + uuid_box)
        output.write_bytes(b"previous output")

        def fail_patch(*_args: object, **_kwargs: object) -> None:
            raise OSError("synthetic patch failure")

        monkeypatch.setattr(isobmff, "_overwrite_range", fail_patch)

        with pytest.raises(OSError, match="synthetic patch failure"):
            isobmff.strip_isobmff_media_file(source, output)

        assert output.read_bytes() == b"previous output"
        assert not list(tmp_path.glob(".clean-*"))


class TestIterTopLevelBoxes:
    """The box walker's three size encodings and its underflow/overflow guards."""

    def test_64bit_largesize(self):
        from remove_ai_watermarks._internal.isobmff import _iter_top_level_boxes

        # size32 == 1 -> a 64-bit largesize follows the type; total box length = 24.
        box = struct.pack(">I", 1) + b"uuid" + struct.pack(">Q", 24) + b"payload!"
        boxes = list(_iter_top_level_boxes(box))
        assert len(boxes) == 1
        start, end, btype, payload_off = boxes[0]
        assert (start, end, btype, payload_off) == (0, 24, b"uuid", 16)

    def test_size0_runs_to_eof(self):
        from remove_ai_watermarks._internal.isobmff import _iter_top_level_boxes

        box = struct.pack(">I", 0) + b"mdat" + b"tail-to-eof"
        boxes = list(_iter_top_level_boxes(box))
        assert len(boxes) == 1
        start, end, btype, payload_off = boxes[0]
        assert (start, end, btype, payload_off) == (0, len(box), b"mdat", 8)

    def test_underflow_size_stops_safely(self):
        from remove_ai_watermarks._internal.isobmff import _iter_top_level_boxes

        # size (4) < the 8-byte header -> the guard returns without yielding a box.
        assert list(_iter_top_level_boxes(struct.pack(">I", 4) + b"ftyp" + b"more")) == []

    def test_overflow_size_stops_safely(self):
        from remove_ai_watermarks._internal.isobmff import _iter_top_level_boxes

        # size claims 999 but the buffer is far shorter -> guard returns, no partial box.
        assert list(_iter_top_level_boxes(struct.pack(">I", 999) + b"uuid" + b"x")) == []


class TestBlankAiXmpPackets:
    """XMP-packet blanking: same-length overwrite only for AI-marked packets, and only
    when the packet is fully delimited."""

    AIMARK = b"trainedAlgorithmicMedia"

    def test_ai_packet_blanked_same_length(self):
        from remove_ai_watermarks._internal.isobmff import blank_ai_xmp_packets

        packet = b'<?xpacket begin="x"?><x:xmpmeta>' + self.AIMARK + b'</x:xmpmeta><?xpacket end="w"?>'
        data = b"boxhdr" + packet + b"tail"
        out, n = blank_ai_xmp_packets(data)
        assert n == 1
        assert len(out) == len(data)  # same length -> iloc offsets stay valid
        assert self.AIMARK not in out
        assert b"boxhdr" in out
        assert b"tail" in out

    def test_clean_packet_left_intact(self):
        from remove_ai_watermarks._internal.isobmff import blank_ai_xmp_packets

        packet = b'<?xpacket begin="x"?><x:xmpmeta>plain copyright</x:xmpmeta><?xpacket end="w"?>'
        out, n = blank_ai_xmp_packets(packet)
        assert n == 0
        assert out == packet

    def test_missing_end_delimiter_not_blanked(self):
        from remove_ai_watermarks._internal.isobmff import blank_ai_xmp_packets

        # No <?xpacket end?> -> the packet regex cannot match, so it is left unchanged.
        data = b'<?xpacket begin="x"?><x:xmpmeta>' + self.AIMARK + b"</x:xmpmeta>"
        out, n = blank_ai_xmp_packets(data)
        assert n == 0
        assert out == data


class TestC2paBufferScans:
    """The shared buffer-scan helpers (used by both the PNG caBX parser and the
    format-agnostic binary scan). Data-driven off the registries so they stay valid
    as vendors are added."""

    def test_soft_binding_vendors_in(self):
        from remove_ai_watermarks._internal.c2pa import C2PA_SOFT_BINDINGS, soft_binding_vendors_in

        sig, name = next(iter(C2PA_SOFT_BINDINGS.items()))
        assert name in soft_binding_vendors_in(b'...manifest:"' + sig + b'"...tail...')
        assert soft_binding_vendors_in(b"") == []
        assert soft_binding_vendors_in(b"no soft-binding assertion here") == []

    def test_synthid_evidence_requires_openai_watermark_action_but_not_google_action(self):
        from remove_ai_watermarks._internal.c2pa import synthid_evidence_vendors_in

        assert synthid_evidence_vendors_in(b"c2pa OpenAI trainedAlgorithmicMedia") == []
        assert synthid_evidence_vendors_in(b"c2pa OpenAI trainedAlgorithmicMedia c2pa.watermarked.unbound") == [
            "OpenAI"
        ]
        assert synthid_evidence_vendors_in(b"c2pa Google trainedAlgorithmicMedia") == ["Google LLC"]

    @pytest.mark.parametrize(
        ("descriptions", "expected"),
        [([], ["Google LLC"]), (["Added imperceptible SynthID watermark"], ["Google LLC"])],
    )
    def test_structured_google_photos_edit_reads_the_synthid_action(self, descriptions, expected):
        actions = [{"action": "c2pa.deleted", "digitalSourceType": "compositeWithTrainedAlgorithmicMedia"}]
        actions += [{"action": "c2pa.edited", "description": text} for text in descriptions]
        store = {
            "active_manifest": "edit",
            "manifests": {
                "edit": {
                    "signature_info": {"issuer": "Google LLC", "common_name": "Google Photos"},
                    "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
                }
            },
        }

        assert c2pa_info_from_manifest_store(store).get("synthid_vendors") == expected

    def test_google_photos_ai_edit_establishes_synthid_without_an_action(self):
        """Byte-scan path: Google's checker found SynthID on every Photos AI edit tested."""
        from remove_ai_watermarks._internal.c2pa import synthid_evidence_vendors_in

        edit = b"c2pa Google LLC Google Photos c2pa.deleted compositeWithTrainedAlgorithmicMedia"
        assert synthid_evidence_vendors_in(edit) == ["Google LLC"]
        assert synthid_evidence_vendors_in(edit + b" Added imperceptible SynthID watermark") == ["Google LLC"]
        # An OpenAI generation edited in Photos keeps its own action-based evidence.
        assert synthid_evidence_vendors_in(edit + b" OpenAI c2pa.watermarked.unbound") == ["Google LLC", "OpenAI"]

    def test_youtube_reencode_needs_a_recorded_synthid_action(self):
        """YouTube re-signs every upload as Google LLC; only a SynthID action in the chain counts."""
        from remove_ai_watermarks._internal.c2pa import synthid_evidence_vendors_in

        upload = b"c2pa Google LLC YouTube Video Processing Services c2pa.transcoded trainedAlgorithmicMedia"
        assert synthid_evidence_vendors_in(upload) == []
        assert synthid_evidence_vendors_in(upload + b" Added imperceptible SynthID watermark") == ["Google LLC"]

    def test_fingerprint_does_not_suppress_fallback_google_synthid(self):
        from remove_ai_watermarks._internal.c2pa import _populate_registry_fields

        info: dict[str, object] = {}
        _populate_registry_fields(b"c2pa Google trainedAlgorithmicMedia io.iscc.v0", info)

        assert info["soft_binding"] == "ISCC (content code)"
        assert info["synthid_vendors"] == ["Google LLC"]

    def test_synthid_verdict_format(self):
        from remove_ai_watermarks._internal.c2pa import synthid_verdict

        assert synthid_verdict("Google LLC") == "present according to Google LLC provenance"


def _amf0_str(value: bytes, *, long: bool = False) -> bytes:
    marker = b"\x0c" if long else b"\x02"
    return marker + len(value).to_bytes(4 if long else 2, "big") + value


def _amf0_property(name: bytes, value: bytes) -> bytes:
    return len(name).to_bytes(2, "big") + name + value


_AMF0_OBJECT_END = b"\x00\x00\x09"

# A minimal TC260-PG-20257A label: the reader validates the JSON before accepting it,
# so the walker tests need a value that actually parses.
_TC260_AIGC_VALUE = (
    b'{"Label":"1","ContentProducer":"00119144030008867405X210002",'
    b'"ProduceID":"sample-001","ReservedCode1":"","ContentPropagator":"",'
    b'"PropagateID":"","ReservedCode2":""}'
)


class TestFlvAmf0Walker:
    """``_skip_amf0`` is what lets the FLV reader step over every property that is not
    ``AIGC``. Each AMF0 type it does not walk correctly aborts the scan, so a label that
    sits after an unhandled type is silently missed. Pure byte parsing -- no media file
    and no decoder is involved, so every branch is reachable from synthetic bytes."""

    @pytest.mark.parametrize(
        ("name", "encoded"),
        [
            ("number", b"\x00" + b"\x00" * 8),
            ("boolean", b"\x01\x01"),
            ("string", _amf0_str(b"a string")),
            ("null", b"\x05"),
            ("undefined", b"\x06"),
            ("reference", b"\x07\x00\x01"),
            ("date", b"\x0b" + b"\x00" * 10),
            ("long-string", _amf0_str(b"a long string", long=True)),
            ("strict-array", b"\x0a\x00\x00\x00\x02" + b"\x00" + b"\x00" * 8 + b"\x01\x00"),
            ("object", b"\x03" + _amf0_property(b"inner", b"\x01\x00") + _AMF0_OBJECT_END),
            ("ecma-array", b"\x08\x00\x00\x00\x01" + _amf0_property(b"inner", b"\x05") + _AMF0_OBJECT_END),
        ],
    )
    def test_every_walkable_type_is_stepped_over(self, name: str, encoded: bytes):
        """A property of this type, sitting before the AIGC one, must not stop the walk."""
        from remove_ai_watermarks._internal.flv import _script_payloads

        payload = (
            _amf0_str(b"onMetaData")
            + b"\x03"
            + _amf0_property(name.encode(), encoded)
            + _amf0_property(b"AIGC", _amf0_str(_TC260_AIGC_VALUE))
            + _AMF0_OBJECT_END
        )
        assert _script_payloads(payload) == (_TC260_AIGC_VALUE,)

    def test_unknown_type_marker_stops_the_walk(self):
        """An unrecognized marker has an unknown width, so the reader cannot guess where
        the next property starts. It must give up rather than resynchronize on garbage."""
        from remove_ai_watermarks._internal.flv import _script_payloads

        payload = (
            _amf0_str(b"onMetaData")
            + b"\x03"
            + _amf0_property(b"mystery", b"\x7f")
            + _amf0_property(b"AIGC", _amf0_str(_TC260_AIGC_VALUE))
            + _AMF0_OBJECT_END
        )
        assert _script_payloads(payload) == ()

    def test_truncated_value_stops_the_walk(self):
        """A declared length running past the buffer must return empty, not raise."""
        from remove_ai_watermarks._internal.flv import _script_payloads

        payload = _amf0_str(b"onMetaData") + b"\x03" + _amf0_property(b"trunc", b"\x02\x00\xff")
        assert _script_payloads(payload) == ()

    def test_nesting_deeper_than_the_depth_cap_is_refused(self):
        """The depth cap bounds work on hostile input; past it the walker returns None."""
        from remove_ai_watermarks._internal.flv import _skip_amf0

        nested = b"\x05"
        for _ in range(12):
            nested = b"\x03" + _amf0_property(b"n", nested) + _AMF0_OBJECT_END
        assert _skip_amf0(nested, 0) is None

    def test_long_string_aigc_value_is_read(self):
        """TC260 values large enough to need the 4-byte long-string form still parse."""
        from remove_ai_watermarks._internal.flv import _script_payloads

        payload = (
            _amf0_str(b"onMetaData")
            + b"\x03"
            + _amf0_property(b"AIGC", _amf0_str(_TC260_AIGC_VALUE, long=True))
            + _AMF0_OBJECT_END
        )
        assert _script_payloads(payload) == (_TC260_AIGC_VALUE,)

    def test_non_onmetadata_script_tag_is_ignored(self):
        """Only ``onMetaData`` carries the normative label; other script tags are skipped."""
        from remove_ai_watermarks._internal.flv import _script_payloads

        payload = (
            _amf0_str(b"onCuePoint")
            + b"\x03"
            + _amf0_property(b"AIGC", _amf0_str(_TC260_AIGC_VALUE))
            + _AMF0_OBJECT_END
        )
        assert _script_payloads(payload) == ()

    def test_missing_file_reads_as_no_payloads(self, tmp_path: Path):
        from remove_ai_watermarks._internal.flv import tc260_aigc_payloads

        assert tc260_aigc_payloads(tmp_path / "absent.flv") == ()

    def test_non_flv_signature_reads_as_no_payloads(self, tmp_path: Path):
        from remove_ai_watermarks._internal.flv import tc260_aigc_payloads

        path = tmp_path / "fake.flv"
        path.write_bytes(b"NOTFLV\x00\x00\x09" + b"\x00" * 32)
        assert tc260_aigc_payloads(path) == ()


class TestProbeMemoization:
    """The per-file probes are cached on (path, mtime_ns, size).

    Their only real failure mode is staleness after an IN-PLACE rewrite, which this
    package does (``remove_ai_metadata(p, p)``, and the batch case where the output
    directory is the input directory). mtime alone can land inside one tick on a
    coarse filesystem, hence size in the key too.
    """

    def test_in_place_strip_invalidates_the_label_cache(self, tmp_path: Path):
        import shutil

        from remove_ai_watermarks.metadata import aigc_label, remove_ai_metadata

        source = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "doubao-1.png"
        if not source.exists():
            pytest.skip("doubao sample not present")
        target = tmp_path / "in_place.png"
        shutil.copyfile(source, target)

        assert aigc_label(target) is not None  # populates the cache
        remove_ai_metadata(target, target)
        assert aigc_label(target) is None, "the cache answered from the pre-strip content"

    def test_caller_cannot_mutate_the_cached_label(self, tmp_path: Path):
        """``aigc_label`` returns a dict; a caller editing it must not poison the cache."""
        import shutil

        from remove_ai_watermarks.metadata import aigc_label

        source = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "doubao-1.png"
        if not source.exists():
            pytest.skip("doubao sample not present")
        target = tmp_path / "mutate.png"
        shutil.copyfile(source, target)

        first = aigc_label(target)
        assert first is not None
        first["ContentProducer"] = "TAMPERED"
        second = aigc_label(target)
        assert second is not None
        assert second["ContentProducer"] != "TAMPERED"

    def test_unstattable_path_bypasses_the_cache_and_behaves_as_before(self, tmp_path: Path):
        """A path that cannot be stat'ed has no cache key, so it must fall through to the
        uncached implementation -- same outcome as before memoization, whatever that is."""
        from remove_ai_watermarks import metadata

        missing = tmp_path / "absent.png"
        assert metadata._stat_key(missing) is None

        def outcome(fn):
            try:
                return ("value", fn(missing))
            except Exception as exc:
                return ("raised", type(exc).__name__)

        assert outcome(metadata.aigc_label) == outcome(metadata._aigc_label_impl)


class TestC2PAInvalidSignature:
    """A .png file that is not actually PNG-signed must read as clean, not crash."""

    def test_has_c2pa_false_for_non_png_bytes(self, tmp_path: Path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"\xff\xd8\xff\xe0 not a png at all, just garbage bytes")
        assert has_c2pa_metadata(fake) is False

    def test_extract_chunk_none_for_non_png_bytes(self, tmp_path: Path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"\xff\xd8\xff\xe0 not a png at all, just garbage bytes")
        assert extract_c2pa_chunk(fake) is None


class TestTc260ContainerRouting:
    """The native-container readers route on CONTENT, not on the file extension.

    Every reader self-gates on its own magic bytes after a 4-12 byte read, so gating
    the AVI and FLV ones on the suffix as well was redundant -- and it made a
    correctly-formatted container served under the wrong name invisible, contradicting
    this module's own rule that format detection reads the bytes.
    """

    @staticmethod
    def _riff_chunk(chunk_id: bytes, payload: bytes) -> bytes:
        return chunk_id + len(payload).to_bytes(4, "little") + payload + (b"\x00" if len(payload) & 1 else b"")

    def _labeled_avi(self) -> bytes:
        info = self._riff_chunk(b"AIGC", _TC260_AIGC_VALUE)
        body = b"AVI " + self._riff_chunk(b"LIST", b"INFO" + info)
        return b"RIFF" + len(body).to_bytes(4, "little") + body

    def test_a_mislabeled_avi_is_still_read(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        target = tmp_path / "clip.bin"  # correct AVI bytes, wrong suffix
        target.write_bytes(self._labeled_avi())
        label = aigc_label(target)
        assert label is not None
        assert label["Label"] == "1"

    def test_a_correctly_named_avi_still_works(self, tmp_path: Path):
        from remove_ai_watermarks.metadata import aigc_label

        target = tmp_path / "clip.avi"
        target.write_bytes(self._labeled_avi())
        assert aigc_label(target) is not None

    def test_webp_yields_nothing_from_the_riff_reader(self, tmp_path: Path):
        """WebP is the one input class the now-unconditional RIFF reader newly touches,
        and it shares the ``RIFF`` prefix -- the ``AVI `` form check is what rejects it."""
        from remove_ai_watermarks._internal.riff import tc260_aigc_payloads

        body = b"WEBP" + self._riff_chunk(b"VP8L", b"\x00" * 16)
        target = tmp_path / "pic.webp"
        target.write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)
        assert tc260_aigc_payloads(target) == ()

    def test_every_reader_is_reached_in_a_stable_order(self):
        from remove_ai_watermarks.metadata import _tc260_container_readers

        readers = _tc260_container_readers()
        assert [r.__module__.rsplit(".", 1)[-1] for r in readers] == ["isobmff", "ebml", "riff", "flv"]


class TestC2paReaderFailureIsVisible:
    """A reader failure and a file with no manifest both return None, so the caller
    cannot tell them apart -- and the consequence is not symmetric. A file with no
    manifest is a normal verdict; a reader that could not read a file it was handed
    can silently downgrade one, so the log level must make the failure observable."""

    def _records(self, caplog, path: str) -> list[str]:
        from remove_ai_watermarks._internal import c2pa

        with caplog.at_level(logging.DEBUG, logger="remove_ai_watermarks._internal.c2pa"):
            assert c2pa._manifest_json_uncached(path) is None
        return [f"{r.levelname} {r.getMessage()}" for r in caplog.records]

    def test_an_unreadable_file_warns(self, caplog):
        records = self._records(caplog, "/nonexistent/definitely-not-here.png")

        assert any(r.startswith("WARNING") for r in records), records

    def test_an_unsupported_container_stays_quiet(self, caplog, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("plain text, not a container the reader handles")

        records = self._records(caplog, str(target))

        assert not any(r.startswith("WARNING") for r in records), records

    def test_a_plain_image_without_a_manifest_logs_nothing(self, caplog, tmp_path: Path):
        target = tmp_path / "plain.png"
        Image.new("RGB", (8, 8)).save(target)

        records = self._records(caplog, str(target))

        assert records == []


class TestC2pa24Assertions:
    """C2PA 2.4 AI disclosure and ingredient digitalSourceType (spec 2.4, April 2026)."""

    @staticmethod
    def _store(assertions: list[dict], ingredients: list[dict] | None = None) -> dict:
        manifest: dict = {"claim_generator": "some-editor/1.0", "assertions": assertions}
        if ingredients is not None:
            manifest["ingredients"] = ingredients
        return {"active_manifest": "active", "manifests": {"active": manifest}}

    @pytest.mark.parametrize(
        ("oversight", "kind"),
        [("fully_autonomous", "generated"), ("prompt_guided", "generated"), ("human_validated", "enhanced")],
    )
    def test_ai_disclosure_alone_asserts_ai_with_the_spec_kind(self, oversight: str, kind: str):
        store = self._store(
            [
                {
                    "label": "c2pa.ai-disclosure",
                    "data": {
                        "modelType": "diffusion",
                        "modelName": "Example Diffusion 3",
                        "contentProfile": {"humanOversightLevel": oversight},
                    },
                }
            ]
        )

        info = c2pa_info_from_manifest_store(store)

        assert info["ai_source_kind"] == kind
        assert info["ai_disclosure"] == "Example Diffusion 3"
        assert info["human_oversight"] == oversight

    def test_digital_source_type_still_decides_the_kind_over_disclosure(self):
        store = self._store(
            [
                {
                    "label": "c2pa.actions.v2",
                    "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": "trainedAlgorithmicMedia"}]},
                },
                {
                    "label": "c2pa.ai-disclosure",
                    "data": {"modelType": "diffusion", "contentProfile": {"humanOversightLevel": "human_validated"}},
                },
            ]
        )

        assert c2pa_info_from_manifest_store(store)["ai_source_kind"] == "generated"

    def test_ingredient_digital_source_type_is_read(self):
        store = self._store(
            [],
            ingredients=[
                {
                    "title": "background.png",
                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                }
            ],
        )

        assert c2pa_info_from_manifest_store(store)["ai_source_kind"] == "generated"

    def test_manifest_without_either_signal_stays_unclaimed(self):
        assert "ai_source_kind" not in c2pa_info_from_manifest_store(self._store([]))


def test_byte_fallback_reads_ai_disclosure_label():
    from remove_ai_watermarks._internal.c2pa import _populate_registry_fields

    info: dict = {}
    assert _populate_registry_fields(b"jumb c2pa c2pa.ai-disclosure modelType human_validated", info) is True
    assert info["ai_source_kind"] == "enhanced"

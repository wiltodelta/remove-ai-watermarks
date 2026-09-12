"""Tests for the provenance identifier (identify.py).

Pure attribution logic is unit-tested directly; end-to-end verdicts assert
against the real committed C2PA / IPTC fixtures in data/fixtures/provenance/.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from remove_ai_watermarks._internal.c2pa import c2pa_info_from_manifest_store
from remove_ai_watermarks._internal.constants import (
    C2PA_AI_VENDORS,
    C2PA_CLAIM_GENERATOR_PLATFORMS,
    C2PA_IDENTITY_AI_ORGS,
)
from remove_ai_watermarks.identify import (
    ProvenanceEvidence,
    ProvenanceReport,
    _ai_tools_in,
    _attribute_platform,
    _integrity_clashes,
    _issuers_in,
    _tc260_manufacturer_of,
    _vendor_of,
    evidence_from_metadata_record,
    extract_provenance_evidence,
    has_invisible_target,
    identify,
    identify_from_evidence,
)
from remove_ai_watermarks.watermark_registry import GEMINI_SPARKLE_TRUST_CONF

# Where the lazy import inside identify._visible_sparkle resolves the detector.
_SPARKLE_TARGET = "remove_ai_watermarks.gemini_engine.detect_sparkle_confidence"

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "provenance"


class TestProvenanceEvidence:
    def test_exact_ai_claim_generator_can_assert_ai_without_source_type(self, tmp_path: Path):
        path = tmp_path / "firefly.png"
        info = {
            "has_c2pa": True,
            "issuer": "Adobe",
            "claim_generator": "Adobe_Firefly",
            "ai_tool": "Firefly",
            "c2pa_identity_ai": True,
            "c2pa_validation_source": "reader",
            "c2pa_validation_state": "Valid",
            "c2pa_integrity": "valid",
            "c2pa_signature": "valid",
            "c2pa_signer_trust": "untrusted",
            "c2pa_signer_validity": "valid",
            "c2pa_validation_codes": ["assertion.dataHash.match", "claimSignature.validated"],
        }
        evidence = ProvenanceEvidence(
            path=path,
            c2pa_info=info,
            ai_metadata={},
            scan=b"jumb c2pa Adobe_Firefly",
            iptc_ai_system=None,
            aigc_label=None,
            exif_generator=None,
            xai_signature=False,
            huggingface_job=None,
            samsung_genai=None,
        )

        report = identify_from_evidence(evidence)

        assert report.is_ai_generated is True
        # Intact binding and signature. The signer is not anchored, which is a missing
        # input here (no trust bundle ships), not a finding against the credential.
        assert report.confidence == "high"
        assert report.platform == "Adobe Firefly"

    def test_revoked_signing_credential_is_disqualifying(self, tmp_path: Path):
        """A credential the issuer disowned cannot establish origin.

        Revocation arrives on its own dimension, not as a binding or signature failure,
        so a check that reads only those two returned an AI verdict off a dead cert with
        an empty ``integrity_clashes`` -- quieter than a hash mismatch on the same file.

        The evidence comes from :func:`c2pa_info_from_manifest_store`, not a hand-written
        dict of what it is believed to emit, so the assertion follows the producer when
        its contract changes.
        """
        path = tmp_path / "revoked.png"
        info = c2pa_info_from_manifest_store(
            {
                "active_manifest": "created",
                "validation_results": {
                    "activeManifest": {
                        "success": [
                            {"code": "assertion.dataHash.match"},
                            {"code": "claimSignature.validated"},
                        ],
                        "failure": [{"code": "signingCredential.ocsp.revoked"}],
                    }
                },
                "manifests": {
                    "created": {
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
                    }
                },
            }
        )
        assert info["c2pa_signer_validity"] == "invalid"
        evidence = ProvenanceEvidence(
            path=path,
            c2pa_info=info,
            ai_metadata={},
            scan=b"jumb c2pa OpenAI trainedAlgorithmicMedia",
            iptc_ai_system=None,
            aigc_label=None,
            exif_generator=None,
            xai_signature=False,
            huggingface_job=None,
            samsung_genai=None,
        )

        report = identify_from_evidence(evidence)

        assert report.is_ai_generated is None
        assert report.platform is None
        assert report.confidence == "none"
        assert any("revoked" in clash for clash in report.integrity_clashes)

    def test_anchored_signer_is_also_high_confidence(self, tmp_path: Path):
        path = tmp_path / "validated.png"
        info = {
            "has_c2pa": True,
            "issuer": "OpenAI",
            "source_type": "trainedAlgorithmicMedia (AI-generated)",
            "ai_source_kind": "generated",
            "c2pa_validation_source": "reader",
            "c2pa_validation_state": "Trusted",
            "c2pa_integrity": "valid",
            "c2pa_signature": "valid",
            "c2pa_signer_trust": "trusted",
            "c2pa_signer_validity": "valid",
            "c2pa_validation_codes": [
                "assertion.dataHash.match",
                "claimSignature.validated",
                "signingCredential.trusted",
            ],
        }
        evidence = ProvenanceEvidence(
            path=path,
            c2pa_info=info,
            ai_metadata={},
            scan=b"jumb c2pa OpenAI trainedAlgorithmicMedia",
            iptc_ai_system=None,
            aigc_label=None,
            exif_generator=None,
            xai_signature=False,
            huggingface_job=None,
            samsung_genai=None,
        )

        report = identify_from_evidence(evidence)

        assert report.is_ai_generated is True
        assert report.confidence == "high"
        assert report.platform == "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)"
        assert not any("not anchored" in caveat for caveat in report.caveats)

    def test_external_metadata_record_builds_equivalent_evidence(self, tmp_path: Path):
        path = tmp_path / "external.jpg"
        signature = "A" * 64
        artist = "c8045292-06d2-4c7d-b4f0-4f93b94e4801"
        record = {
            "pil": {"info:parameters": "Steps: 20, Sampler: Euler"},
            "exif": {
                "0th": {
                    "ImageDescription": f"Signature: {signature}",
                    "Artist": artist,
                }
            },
        }

        evidence = evidence_from_metadata_record(record, path=path)
        report = identify_from_evidence(evidence)

        assert evidence.path == path
        assert evidence.ai_metadata["parameters"] == "Steps: 20, Sampler: Euler"
        assert evidence.xai_signature is True
        assert report.is_ai_generated is True
        assert {signal.name for signal in report.signals} >= {"gen_params", "xai_signature"}

    def test_external_scanner_diagnostics_do_not_create_c2pa_evidence(self, tmp_path: Path):
        path = tmp_path / "plain.jpg"
        record = {
            "c2pa_store": {"error": "ManifestNotFound: no JUMBF data found"},
            "jpeg": {
                "segments": [
                    {
                        "marker": "APP11",
                        "kind": "c2pa_or_jumbf",
                        "base64": "AAA=",
                    }
                ]
            },
        }

        report = identify_from_evidence(evidence_from_metadata_record(record, path=path))

        assert report.is_ai_generated is None
        assert report.signals == []
        assert report.watermarks == []

    def test_external_scanner_raw_bytes_still_create_c2pa_evidence(self, tmp_path: Path):
        path = tmp_path / "signed.jpg"
        manifest = b"jumb c2pa OpenAI trainedAlgorithmicMedia"
        record = {
            "jpeg": {
                "segments": [
                    {
                        "marker": "APP11",
                        "kind": "c2pa_or_jumbf",
                        "base64": base64.b64encode(manifest).decode(),
                    }
                ]
            }
        }

        report = identify_from_evidence(evidence_from_metadata_record(record, path=path))

        assert report.is_ai_generated is True
        assert report.platform == "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)"
        assert [signal.name for signal in report.signals] == ["c2pa"]

    def test_external_generator_bytes_are_normalized(self, tmp_path: Path):
        evidence = evidence_from_metadata_record(
            {"exif": {"0th": {"Software": b"NovelAI"}}},
            path=tmp_path / "external.png",
        )

        assert evidence.exif_generator == "NovelAI"

    @pytest.mark.parametrize(
        "record",
        [
            {"name": "trainedAlgorithmicMedia.jpg"},
            {"sha256": "jumb-c2pa-OpenAI-trainedAlgorithmicMedia"},
            {"pil": {"trainedAlgorithmicMedia": "plain"}},
            {"signals": {"provenance": {"is_ai_generated": True}}},
            {"pixel": {"error": "trainedAlgorithmicMedia"}},
        ],
    )
    def test_external_diagnostics_and_arbitrary_keys_are_not_evidence(self, tmp_path: Path, record: dict):
        report = identify_from_evidence(evidence_from_metadata_record(record, path=tmp_path / "plain.jpg"))

        assert report.is_ai_generated is None
        assert report.signals == []

    def test_external_metadata_value_is_evidence(self, tmp_path: Path):
        record = {
            "exif": {
                "0th": {
                    "ImageDescription": "digitalSourceType=trainedAlgorithmicMedia",
                }
            }
        }

        report = identify_from_evidence(evidence_from_metadata_record(record, path=tmp_path / "generated.jpg"))

        assert report.is_ai_generated is True
        assert report.ai_source_kind == "generated"

    @pytest.mark.parametrize(
        "filename",
        [
            "chatgpt-1.png",
            "chatgpt-2.png",
            "doubao-1.png",
            "firefly-1.png",
            "flux-1.jpg",
            "flux-1.png",
            "grok-1.jpg",
            "mj-1.png",
        ],
    )
    def test_metadata_only_identify_matches_extracted_evidence(self, filename: str):
        path = SAMPLES_DIR / filename

        direct = identify(path, check_visible=False, check_invisible=False)
        evidence = extract_provenance_evidence(path)
        extracted = identify_from_evidence(evidence)

        assert isinstance(evidence, ProvenanceEvidence)
        assert extracted == direct

    def test_identify_from_evidence_does_not_read_the_source(self, monkeypatch, tmp_path: Path):
        path = tmp_path / "generated.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2paOpenAI DALL-E trainedAlgorithmicMedia\xff\xd9")
        evidence = extract_provenance_evidence(path)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("identify_from_evidence must not read the source file")

        monkeypatch.setattr("remove_ai_watermarks.identify.extract_c2pa_info", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.get_ai_metadata", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.scan_head", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.iptc_ai_system", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.aigc_label", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.exif_generator", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.xai_signature", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.huggingface_job", fail_if_called)
        monkeypatch.setattr("remove_ai_watermarks.identify.samsung_genai", fail_if_called)
        monkeypatch.setattr("builtins.open", fail_if_called)
        monkeypatch.setattr(Path, "open", fail_if_called)

        report = identify_from_evidence(evidence)

        assert report.is_ai_generated is True
        assert any(signal.name == "c2pa" for signal in report.signals)


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

    def test_canva(self):
        platform = _attribute_platform(["Canva"])
        assert platform
        assert "Canva" in platform

    def test_byteplus_keeps_its_product_name(self):
        # ByteDance's intl brand signs as "Byteplus Pte. Ltd."; the registry maps
        # it to the ByteDance family (was mis-read as Adobe via an incidental
        # "Adobe XMP" file string before the entry existed).
        platform = _attribute_platform(["BytePlus (ByteDance)"])
        assert platform == "BytePlus (ByteDance)"

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
        path = self._c2pa_jpeg(
            tmp_path,
            b"OpenAI DALL-E ... trainedAlgorithmicMedia ... c2pa.watermarked.unbound",
        )
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
        assert r.platform == "ByteDance Volcano Engine"

    def test_bytedance_chinese_legal_name_attributed(self, tmp_path: Path):
        # Some Volcano Engine certs name the signer with the Chinese legal entity
        # rather than the latin "volcengine"; the latin needle misses it, so the
        # Chinese-name registry entry is what attributes real ByteDance output.
        blob = "北京火山引擎科技有限公司".encode() + b" ... trainedAlgorithmicMedia"
        path = self._c2pa_jpeg(tmp_path, blob)
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "ByteDance Volcano Engine"

    @pytest.mark.parametrize(
        ("claim_generator", "platform"),
        [
            ("Grok Imagine", "xAI Grok Imagine"),
            ("Suno", "Suno"),
            ("Higgsfield AI", "Higgsfield AI"),
            ("recraft.ai", "Recraft"),
            ("Topaz Labs Image API", "Topaz Labs"),
            ("TIKTOK AD Creative Toolbox", "TikTok Ad Creative Toolbox"),
        ],
    )
    def test_claim_generator_wins_over_upstream_issuer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claim_generator: str, platform: str
    ):
        path = tmp_path / "generated.png"
        from PIL import Image

        Image.new("RGB", (32, 32)).save(path)
        monkeypatch.setattr(
            "remove_ai_watermarks.identify.extract_c2pa_info",
            lambda _path: {
                "has_c2pa": True,
                "issuer": "OpenAI",
                "claim_generator": claim_generator,
                "source_type": "trainedAlgorithmicMedia (AI-generated)",
                "ai_source_kind": "generated",
            },
        )

        report = identify(path, check_visible=False, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.platform == platform
        assert claim_generator in next(s.detail for s in report.signals if s.name == "c2pa")

    def test_real_recraft_c2pa_fixture(self):
        report = identify(SAMPLES_DIR / "recraft-v3.webp", check_visible=False)

        assert report.is_ai_generated is True
        assert report.confidence == "high"
        assert report.platform == "Recraft"
        assert any(signal.name == "c2pa" and "recraft.ai" in signal.detail for signal in report.signals)

    def test_real_krea_api_fixture_has_no_local_provenance(self):
        report = identify(SAMPLES_DIR / "krea-2-medium-turbo.png", check_visible=False)

        assert report.is_ai_generated is None
        assert report.platform is None
        assert report.signals == []

    def test_real_direct_kling_fixture_has_tc260_and_current_visible_mark(self):
        path = SAMPLES_DIR.parent / "visible" / "kling" / "provider-original-direct.png"
        report = identify(path, check_visible=True)

        assert report.is_ai_generated is True
        assert any(
            signal.name == "aigc" and "001191110108335469089C10100" in signal.detail for signal in report.signals
        )
        assert any(signal.name == "visible_kling" for signal in report.signals)

    @pytest.mark.parametrize("filename", ["qwen-image.png", "seedream-v4.jpg"])
    def test_real_wavespeed_api_fixture_has_no_local_provenance(self, filename):
        report = identify(SAMPLES_DIR / filename, check_visible=False)

        assert report.is_ai_generated is None
        assert report.platform is None
        assert report.signals == []

    def test_real_wavespeed_hunyuan_fixture_has_fal_c2pa(self):
        path = SAMPLES_DIR / "hunyuan-image-3.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            "6ae2ac073ac79b02b0934c905364f200420c2ff60fea232bd9e0a5fca785cc09"
        )

        report = identify(path, check_visible=False, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.platform == "fal.ai"
        assert report.confidence == "high"
        assert any(signal.name == "c2pa" and "fal-ai/hunyuan-image" in signal.detail for signal in report.signals)

    def test_real_wavespeed_kling_fixture_has_tc260_metadata(self):
        path = SAMPLES_DIR / "kling-image-v3.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            "501ab865b47ba59b64bc0f68118300744b0d5c7be2fa4e2ecbb3a78de5d8a82b"
        )

        report = identify(path, check_visible=False, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.confidence == "high"
        aigc = next(signal for signal in report.signals if signal.name == "aigc")
        assert "001191110108335469089C10100" in aigc.detail

    def test_real_qwen_create_fixture_has_tc260_metadata(self):
        path = SAMPLES_DIR / "qwen-create-qwen-image-2.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            "60c39ed8aedb081193c62289be549785e74d31fc8d071961003fabd74712429d"
        )

        report = identify(path, check_visible=False, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.confidence == "high"
        aigc = next(signal for signal in report.signals if signal.name == "aigc")
        assert "001191440101MA9Y9T4H7A00000" in aigc.detail

    def test_dreamina_attributed_without_source_type(self, tmp_path: Path):
        # Dreamina (ByteDance's international Jimeng brand) signs C2PA as
        # "Bytedance Pte. Ltd." with a "Dreamina/x.y" claim generator and NO
        # digitalSourceType assertion -- the generator name is the only AI signal.
        # It is an identity-AI vendor (a pure generator), so attribution must not
        # depend on trainedAlgorithmicMedia the way the incidental-mention-prone
        # common-word issuers (Adobe/Google/OpenAI) do.
        path = self._c2pa_jpeg(tmp_path, b"Bytedance Pte. Ltd. Dreamina/7.5.0 c2pa.created")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "ByteDance Dreamina"

    def test_elevenlabs_attributed(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Eleven Labs Inc. ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "ElevenLabs"
        assert not any("SynthID" in w for w in r.watermarks)  # ElevenLabs does not use SynthID

    def test_fal_ai_attributed(self, tmp_path: Path):
        # fal.ai signs as "fal - Features & Labels Inc." with a "fal-ai/<model>"
        # claim generator.
        path = self._c2pa_jpeg(tmp_path, b"fal - Features & Labels Inc. fal-ai/seedvr trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "fal.ai"

    def test_bria_attributed_without_source_type(self, tmp_path: Path):
        # Bria signs as "Bria Artificial Intelligence" with source type ``empty``
        # (NO trainedAlgorithmicMedia) -- a pure-generator asserts_ai vendor, so
        # the issuer/generator strings alone must flag AI.
        path = self._c2pa_jpeg(tmp_path, b"Bria Artificial Intelligence Bria Ai c2pa.created c2pa.edited")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Bria AI"

    def test_stability_ai_issuer_attributed_no_synthid(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Stability AI ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Stability AI" in r.platform
        assert not any("SynthID" in w for w in r.watermarks)  # Stability does not use SynthID

    def test_trained_source_is_generated_kind(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.ai_source_kind == "generated"

    def test_composite_source_is_enhanced_kind(self, tmp_path: Path):
        # compositeWithTrainedAlgorithmicMedia: a real photo with an AI-composited
        # region. Still AI (is_ai True), but the kind must read "enhanced" so a
        # caller can do region-targeted cleaning instead of a full-frame regen.
        path = self._c2pa_jpeg(tmp_path, b"Adobe ... compositeWithTrainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.ai_source_kind == "enhanced"

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

    def test_galaxy_trained_source_is_unverified_ai(self, tmp_path: Path):
        path = self._jpeg(tmp_path, "s25.jpg", b"Samsung Galaxy Galaxy S25 c2pa-rs trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform == "Samsung Galaxy (C2PA)"
        assert r.c2pa_validation is None
        assert any("without cryptographic validation" in caveat for caveat in r.caveats)
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

    def test_galaxy_capture_without_ai_marker_is_not_ai(self, tmp_path: Path):
        # A genuine Galaxy phone capture carries Samsung Galaxy C2PA provenance but
        # NO AI source-type / genAIType. It must stay is_ai=None -- the device cert
        # is authenticity provenance of a real photo, not an AI-generation signal.
        path = self._jpeg(tmp_path, "s25_capture.jpg", b"Samsung Galaxy Galaxy S25 c2pa-rs no ai marker")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is None
        assert r.platform == "Samsung Galaxy (C2PA)"
        assert any("C2PA" in w for w in r.watermarks)


# ── End-to-end verdicts on real fixtures ────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/fixtures/provenance not present")
class TestIdentifyRealSamples:
    def test_openai_chatgpt(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform
        assert "OpenAI" in r.platform
        assert any("C2PA" in w for w in r.watermarks)
        assert not any("SynthID" in w for w in r.watermarks)

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

    def test_apple_clean_up_attributed(self, tmp_path: Path):
        # Apple Photos Clean Up (Apple Intelligence object removal) marks the
        # AI edit via photoshop:Credit next to compositeWithTrainedAlgorithmicMedia
        # -- it must be attributed, not reported as a generic made-with-AI tag.
        # This attribution must survive metadata consolidation.
        p = tmp_path / "apple_cleanup.jpg"
        p.write_bytes(
            b'\xff\xd8\xff\xe1<x:xmpmeta photoshop:Credit="Apple Photos Clean Up" '
            b"Iptc4xmpExt:DigitalSourceType=compositeWithTrainedAlgorithmicMedia></x:xmpmeta>\xff\xd9"
        )
        r = identify(p, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Apple Photos (Clean Up AI edit)"
        assert r.ai_source_kind == "enhanced"
        assert "content_seal" not in [signal.name for signal in r.signals]
        assert not any("Meta" in watermark for watermark in r.watermarks)

    def test_standalone_iptc_composite_synthetic_is_enhanced(self, tmp_path: Path):
        p = tmp_path / "composite.jpg"
        p.write_bytes(
            b'\xff\xd8\xff\xe1<x:xmpmeta Iptc4xmpExt:DigitalSourceType="compositeSynthetic"></x:xmpmeta>\xff\xd9'
        )

        r = identify(p, check_visible=False, check_invisible=False)

        assert r.is_ai_generated is True
        assert r.ai_source_kind == "enhanced"

    def test_standalone_ai_tag_does_not_claim_a_vendor_watermark(self, tmp_path: Path):
        """The shared IPTC standard proves neither Meta nor Content Seal."""
        p = tmp_path / "muse-tag.jpg"
        p.write_bytes(
            b'\xff\xd8\xff\xe1<x:xmpmeta Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"></x:xmpmeta>\xff\xd9'
        )

        r = identify(p, check_visible=False, check_invisible=False)

        names = [s.name for s in r.signals]
        assert "iptc" in names
        assert "content_seal" not in names
        assert r.platform is None
        assert not any("Content Seal" in watermark for watermark in r.watermarks)

    def test_flux_bfl_c2pa_png(self):
        # flux-1.png: real Black Forest Labs FLUX.2 Playground output (signed C2PA).
        r = identify(SAMPLES_DIR / "flux-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Black Forest Labs (FLUX)"

    def test_flux_bfl_c2pa_jpeg_via_reader(self):
        # flux-1.jpg: same source as a JPEG -- the real committed JPEG-with-C2PA
        # fixture that exercises the c2pa-python non-PNG reader path end to end.
        r = identify(SAMPLES_DIR / "flux-1.jpg", check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Black Forest Labs (FLUX)"

    def test_clean_photo_is_unknown_not_clean(self, clean_photo: Path):
        r = identify(clean_photo, check_visible=False)
        assert r.is_ai_generated is None  # never asserted False
        assert r.platform is None
        assert r.confidence == "none"
        assert r.watermarks == []

    def test_has_invisible_target_true_on_metadata_ai(self):
        # The scrub gate: a C2PA/SynthID image and an IPTC "Made with AI" image are
        # both invisible/metadata targets, so the diffusion scrub should run.
        assert has_invisible_target(SAMPLES_DIR / "chatgpt-1.png") is True
        assert has_invisible_target(SAMPLES_DIR / "mj-1.png") is True
        # ai_from_metadata records scrub intent independently of the confidence string.
        assert identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False).ai_from_metadata is True

    def test_untrusted_but_intact_c2pa_is_high_confidence_with_a_caveat(self):
        report = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.confidence == "high"
        assert report.ai_from_metadata is True
        assert report.c2pa_validation is not None
        assert report.c2pa_validation["source"] == "reader"
        assert report.c2pa_validation["state"] == "Invalid"
        assert report.c2pa_validation["integrity"] == "valid"
        assert report.c2pa_validation["signature"] == "valid"
        assert report.c2pa_validation["signer_trust"] == "untrusted"
        assert report.c2pa_validation["signer_validity"] == "expired"
        assert "assertion.dataHash.match" in report.c2pa_validation["codes"]
        # What was not established is said, not folded into the confidence string.
        assert any("never checked against one" in caveat for caveat in report.caveats)
        assert any("only the signing time is unproven" in caveat for caveat in report.caveats)

    def test_no_committed_fixture_reports_a_trusted_signer(self):
        """The reachability guard for :func:`_c2pa_credential_level`.

        The SDK ships no production trust anchors, so ``signingCredential.trusted``
        appears in no default installation. Gating high confidence on it made that branch
        dead in production for every vendor while a hand-built dict kept it green in the
        suite. These fixtures are the producer; if one ever comes back trusted, a bundle
        got configured and the confidence mapping needs re-reading, not this assertion
        deleted.
        """
        checked = 0
        for path in sorted(SAMPLES_DIR.iterdir()):
            report = identify(path, check_visible=False, check_invisible=False)
            if report.c2pa_validation is None:
                continue
            checked += 1
            assert report.c2pa_validation["signer_trust"] != "trusted"
            if report.c2pa_validation["integrity"] == "valid" and report.c2pa_validation["signature"] == "valid":
                assert report.confidence == "high", path.name
        assert checked >= 3

    def test_hash_mismatch_does_not_confirm_origin_but_keeps_scrub_fail_safe(self, tampered_chatgpt_png: Path):
        report = identify(tampered_chatgpt_png, check_visible=False, check_invisible=False)

        assert report.is_ai_generated is None
        assert report.platform is None
        assert report.confidence == "none"
        assert report.ai_source_kind is None
        assert report.ai_from_metadata is False
        assert report.c2pa_validation is not None
        assert report.c2pa_validation["integrity"] == "invalid"
        assert any("dataHash.mismatch" in clash for clash in report.integrity_clashes)
        assert has_invisible_target(tampered_chatgpt_png) is True

    def test_has_invisible_target_false_on_clean_photo(self, clean_photo: Path):
        # No detectable invisible signal -> skip the scrub (do not degrade a clean image).
        assert has_invisible_target(clean_photo) is False
        assert identify(clean_photo, check_visible=False).ai_from_metadata is False

    def test_strip_caveat_always_present(self, clean_photo: Path):
        r = identify(clean_photo, check_visible=False)
        assert any("not proof" in c for c in r.caveats)

    def test_returns_report_dataclass(self):
        assert isinstance(identify(SAMPLES_DIR / "firefly-1.png", check_visible=False), ProvenanceReport)


class TestHasInvisibleTargetFailSafe:
    """The scrub gate fails SAFE: when a detector errors, it runs the removal."""

    def test_detector_error_defaults_to_run(self, tmp_path: Path):
        # If evidence evaluation raises (a detector crash), the gate must return True so the
        # caller still attempts removal -- leaving a watermark on a paid removal is
        # worse than over-regenerating. (Garbage bytes do NOT raise; identify returns
        # a clean None verdict there, so that path correctly skips -- see below.)
        bad = tmp_path / "x.png"
        bad.write_bytes(b"not image bytes")
        with patch("remove_ai_watermarks.identify._identify_from_evidence", side_effect=RuntimeError("boom")):
            assert has_invisible_target(bad) is True

    def test_unreadable_bytes_are_not_a_target(self, tmp_path: Path):
        # No raise, no signal -> not a scrub target (the CLI rejects undecodable
        # images earlier anyway; this only documents the gate's own verdict).
        bad = tmp_path / "x.png"
        bad.write_bytes(b"not image bytes")
        assert has_invisible_target(bad) is False

    def test_local_ai_params_are_a_target(self, tmp_png_with_ai_metadata: Path):
        assert has_invisible_target(tmp_png_with_ai_metadata) is True


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

    def test_local_gen_params_have_no_c2pa_source_kind(self, tmp_png_with_ai_metadata: Path):
        # AI verdict from local SD params (not C2PA) -> ai_source_kind stays None.
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        assert r.is_ai_generated is True
        assert r.ai_source_kind is None

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


# ── Hugging Face-hosted job marker (medium confidence) ─────────────


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
        assert "Hugging Face" in r.platform
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
    """The visible-sparkle signal gates on the calibrated threshold (0.5)."""

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


REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_BEFORE = REPO_ROOT / "demo_banana_before.png"
_DEMO_AFTER = REPO_ROOT / "demo_banana_after.png"


@pytest.mark.skipif(not (_DEMO_BEFORE.exists() and _DEMO_AFTER.exists()), reason="demo banana pair not present")
class TestSparkleDetectRemoveAlignment:
    """Detect (identify) and remove (registry.detect_marks) must agree on the
    same image. A prior desync let identify report a sparkle that
    removal arbitration declined (or vice versa). Both gate on the single shared
    GEMINI_SPARKLE_TRUST_CONF, so a sparkle just over the line is taken by BOTH
    and one just under is declined by BOTH. Fixtures composite the real captured
    sparkle (before-minus-after) back at reduced opacity to land on either side.
    """

    @staticmethod
    def _faint_sparkle(tmp_path: Path, opacity: float) -> Path:
        import numpy as np

        from remove_ai_watermarks import image_io

        before = image_io.imread(_DEMO_BEFORE).astype("float32")
        after = image_io.imread(_DEMO_AFTER).astype("float32")
        faint = np.clip(after + opacity * (before - after), 0, 255).astype("uint8")
        out = tmp_path / f"sparkle_{int(opacity * 100)}.png"
        image_io.imwrite(out, faint)
        return out

    def _detect_remove(self, path: Path) -> tuple[bool, bool, float]:
        from remove_ai_watermarks import image_io, watermark_registry
        from remove_ai_watermarks.gemini_engine import detect_sparkle_confidence

        conf = detect_sparkle_confidence(path) or 0.0
        identify_fires = conf >= GEMINI_SPARKLE_TRUST_CONF
        dets = watermark_registry.detect_marks(image_io.imread(path), include_explicit=False)
        remove_takes_gemini = any(d.key == "gemini" and d.detected for d in dets)
        return identify_fires, remove_takes_gemini, conf

    def test_above_threshold_both_fire(self, tmp_path: Path):
        path = self._faint_sparkle(tmp_path, 0.7)  # ~0.55 conf, just over the line
        identify_fires, remove_takes, conf = self._detect_remove(path)
        assert conf >= GEMINI_SPARKLE_TRUST_CONF
        assert identify_fires, f"identify declined a sparkle above threshold (conf={conf:.3f})"
        assert remove_takes, f"removal declined a sparkle above threshold (conf={conf:.3f})"

    def test_below_threshold_both_decline(self, tmp_path: Path):
        path = self._faint_sparkle(tmp_path, 0.5)  # ~0.37 conf, just under the line
        identify_fires, remove_takes, conf = self._detect_remove(path)
        assert conf < GEMINI_SPARKLE_TRUST_CONF
        assert not identify_fires, f"identify fired below threshold (conf={conf:.3f})"
        assert not remove_takes, f"removal fired below threshold (conf={conf:.3f})"

    def test_full_strength_both_fire(self):
        # The shipped demo sparkle at full strength: unambiguous agreement.
        identify_fires, remove_takes, conf = self._detect_remove(_DEMO_BEFORE)
        assert conf >= GEMINI_SPARKLE_TRUST_CONF
        assert identify_fires
        assert remove_takes


class TestIdentifyImportIsLight:
    """`import identify` must stay torch-free (lazy _internal/__init__): the package
    is deployed on a 512 MB host where eagerly pulling torch/diffusers OOMs."""

    def test_import_identify_does_not_pull_torch(self):
        # Only meaningful where torch is installed (the gpu/detect extra); on a
        # core-only CI runner torch can't be in sys.modules anyway.
        pytest.importorskip("torch")
        code = "import sys, remove_ai_watermarks.identify; sys.exit(1 if 'torch' in sys.modules else 0)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)  # noqa: S603
        assert result.returncode == 0, f"import identify pulled torch: {result.stderr.decode()[-500:]}"


# Where the registry-backed Doubao/Jimeng visible detector resolves.
_TEXT_MARKS_TARGET = "remove_ai_watermarks.identify._visible_text_marks"


class TestIdentifyVisibleTextMarks:
    """The visible Doubao/Jimeng marks are a stripped-metadata visual fallback,
    parallel to the Gemini sparkle: each lifts an Unknown verdict to medium."""

    @staticmethod
    def _detection(key: str, label: str, conf: float):
        from remove_ai_watermarks.watermark_registry import MarkDetection

        return MarkDetection(key, label, "bottom-right", True, conf, (0, 0, 10, 10))

    def test_doubao_promotes_to_medium(self, tmp_clean_png: Path):
        det = self._detection("doubao", "Doubao 豆包AI生成 text", 0.8)
        with patch(_SPARKLE_TARGET, return_value=None), patch(_TEXT_MARKS_TARGET, return_value=[det]):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform is not None
        assert "Doubao" in r.platform
        signal = next(s for s in r.signals if s.name == "visible_doubao")
        assert signal.confidence == "medium"

    def test_jimeng_promotes_to_medium(self, tmp_clean_png: Path):
        det = self._detection("jimeng", "Jimeng 即梦AI wordmark", 0.9)
        with patch(_SPARKLE_TARGET, return_value=None), patch(_TEXT_MARKS_TARGET, return_value=[det]):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform is not None
        assert "Jimeng" in r.platform
        assert any(s.name == "visible_jimeng" for s in r.signals)

    def test_check_visible_false_skips_text_marks(self, tmp_clean_png: Path):
        det = self._detection("doubao", "Doubao 豆包AI生成 text", 0.99)
        with patch(_SPARKLE_TARGET, return_value=None), patch(_TEXT_MARKS_TARGET, return_value=[det]) as mock:
            r = identify(tmp_clean_png, check_visible=False)
        mock.assert_not_called()
        assert not any(s.name == "visible_doubao" for s in r.signals)

    def test_metadata_keeps_high_even_with_text_mark(self, tmp_png_with_ai_metadata: Path):
        det = self._detection("doubao", "Doubao 豆包AI生成 text", 0.8)
        with patch(_SPARKLE_TARGET, return_value=None), patch(_TEXT_MARKS_TARGET, return_value=[det]):
            r = identify(tmp_png_with_ai_metadata, check_visible=True)
        assert r.confidence == "high"

    def test_visible_path_decodes_file_once(self, tmp_clean_png: Path):
        """The web path identify(check_visible=True, check_invisible=False) must
        decode the image exactly once and share the array across the sparkle +
        text-mark detectors. Two decodes of the same bitmap spiked memory on the
        small web worker (the OOM the decode-once refactor addresses).

        Count decodes OF THE SOURCE, not every ``imread`` in the process: the Gemini
        engine loads its own bundled capture assets on first construction, so a
        process-wide count was 3 on a cold engine and 1 on a warm one, and the test
        passed only when some earlier test happened to build the engine first."""
        import remove_ai_watermarks.image_io as image_io

        real_imread = image_io.imread
        with patch.object(image_io, "imread", side_effect=real_imread) as mock_imread:
            identify(tmp_clean_png, check_visible=True, check_invisible=False)
        source_decodes = [c for c in mock_imread.call_args_list if Path(c.args[0]) == Path(tmp_clean_png)]
        assert len(source_decodes) == 1

    def test_missing_pixel_extra_preserves_metadata_verdict(self, tmp_png_with_ai_metadata: Path):
        import remove_ai_watermarks.image_io as image_io

        with patch.object(image_io, "imread", side_effect=ModuleNotFoundError("No module named 'cv2'")):
            report = identify(tmp_png_with_ai_metadata, check_visible=True, check_invisible=False)

        assert report.is_ai_generated is True
        assert report.confidence == "high"


# ── Caveats and serialization ───────────────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/fixtures/provenance not present")
class TestIdentifyCaveats:
    def test_legacy_openai_has_no_synthid_claim(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert not any("SynthID" in watermark for watermark in r.watermarks)
        assert not any("before the rollout" in c for c in r.caveats)

    def test_legacy_openai_has_no_synthid_proxy_caveat(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert not any("not locally" in c for c in r.caveats)

    def test_caveats_are_deduplicated(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert len(r.caveats) == len(set(r.caveats))


class TestSynthIDProvenanceEvidence:
    """Google's provider policy and OpenAI's explicit watermark action are evidence.

    A bare legacy OpenAI C2PA manifest is not: those credentials existed before
    OpenAI adopted SynthID.
    """

    @staticmethod
    def _png_chunk(ctype: bytes, data: bytes) -> bytes:
        import struct
        import zlib

        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

    def _png(self, tmp_path: Path, name: str, *extra: bytes) -> Path:
        import struct
        import zlib

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        body = (
            b"\x89PNG\r\n\x1a\n"
            + self._png_chunk(b"IHDR", ihdr)
            + self._png_chunk(b"IDAT", zlib.compress(b"\x00" * 6, 9))
            + b"".join(extra)
            + self._png_chunk(b"IEND", b"")
        )
        path = tmp_path / name
        path.write_bytes(body)
        return path

    def test_google_synthid_with_incidental_openai_byte_no_caveat(self, tmp_path: Path):
        # Google C2PA/SynthID manifest in caBX; the byte "OpenAI" lives in a
        # separate tEXt chunk (e.g. a trust-chain note), not as a SynthID vendor.
        png = self._png(
            tmp_path,
            "g.png",
            self._png_chunk(b"caBX", b"jumbc2pa Google ... trainedAlgorithmicMedia"),
            self._png_chunk(b"tEXt", b"note\x00signed via OpenAI trust chain"),
        )
        r = identify(png, check_visible=False, check_invisible=False)
        assert any("SynthID watermark (present according to Google" in w for w in r.watermarks)
        assert not any("before the rollout" in c for c in r.caveats)

    def test_openai_watermark_action_asserts_synthid(self, tmp_path: Path):
        png = self._png(
            tmp_path,
            "oa.png",
            self._png_chunk(
                b"caBX",
                b"jumbc2pa OpenAI ... trainedAlgorithmicMedia ... c2pa.watermarked.unbound",
            ),
        )
        r = identify(png, check_visible=False, check_invisible=False)
        assert any("SynthID watermark (present according to OpenAI" in w for w in r.watermarks)
        assert not any("before the rollout" in c for c in r.caveats)

    def test_legacy_openai_without_watermark_action_does_not_assert_synthid(self, tmp_path: Path):
        png = self._png(
            tmp_path, "oa-old.png", self._png_chunk(b"caBX", b"jumbc2pa OpenAI ... trainedAlgorithmicMedia")
        )
        r = identify(png, check_visible=False, check_invisible=False)
        assert not any("SynthID" in watermark for watermark in r.watermarks)

    def test_dreamina_png_cabx_without_source_type(self, tmp_path: Path):
        # Real Dreamina PNGs carry the "Dreamina/x.y" generator in an ingredient
        # manifest (the active one is often a c2pa-tool transcode), so the caBX
        # byte-scan -- which sees the whole store -- is the reliable detector, and
        # there is no trainedAlgorithmicMedia to lean on.
        png = self._png(tmp_path, "dreamina.png", self._png_chunk(b"caBX", b"jumbc2pa Dreamina/7.5.0 c2pa.created"))
        r = identify(png, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.platform == "ByteDance Dreamina"


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
    """C2PA soft-binding algorithms retain their registered kind in the report."""

    def test_soft_binding_vendor_listed(self, tmp_path: Path):
        p = tmp_path / "sb.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe1 c2pa jumb com.digimarc.validate.1 \xff\xd9")
        r = identify(p, check_visible=False, check_invisible=False)
        assert any("Digimarc" in w for w in r.watermarks)
        assert any(s.name == "soft_binding" for s in r.signals)

    def test_fingerprint_is_not_listed_as_a_watermark(self, tmp_path: Path):
        p = tmp_path / "fingerprint.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe1 c2pa jumb io.iscc.v0 \xff\xd9")

        report = identify(p, check_visible=False, check_invisible=False)

        signal = next(signal for signal in report.signals if signal.name == "soft_binding")
        assert signal.detail.startswith("C2PA content fingerprint: ISCC (content code)")
        assert not any("soft binding" in watermark.casefold() for watermark in report.watermarks)
        assert any("fingerprint may still be recomputed" in caveat for caveat in report.caveats)

    def test_invismark_signal_lists_signed_watermark_id(self, tmp_path: Path):
        watermark_id = "83424621-03cb-40e3-9808-a9fae837156d"
        record = {
            "c2pa_store": {
                "active_manifest": "paint",
                "manifests": {
                    "paint": {
                        "assertions": [
                            {
                                "label": "c2pa.soft-binding",
                                "data": {
                                    "alg": "com.microsoft.invismark.1",
                                    "blocks": [{"scope": "the entire image", "value": watermark_id}],
                                },
                            }
                        ]
                    }
                },
            }
        }
        evidence = evidence_from_metadata_record(record, path=tmp_path / "paint.png")

        report = identify_from_evidence(evidence)

        assert evidence.ai_metadata["soft_binding_value"] == watermark_id
        signal = next(signal for signal in report.signals if signal.name == "soft_binding")
        assert "com.microsoft.invismark.1" in signal.detail
        assert watermark_id in signal.detail
        invismark = next(signal for signal in report.signals if signal.name == "invismark")
        assert "com.microsoft.invismark.1" in invismark.detail
        assert watermark_id in invismark.detail
        assert any("may remain in the pixels" in caveat for caveat in report.caveats)

    @pytest.mark.parametrize(
        ("filename", "watermark_id", "sha256"),
        [
            (
                "microsoft-invismark.png",
                "19b070a1-23f0-40eb-826c-cade306337eb",
                "be27ac7c351e7f381dc69e140ca3611ecd9db81e40926c68050ceffdd0ce6c04",
            ),
            (
                "microsoft-paint-invismark.png",
                "a4ed421e-7c37-4409-998e-f8788c564572",
                "74eda22d210d534fcb5b75bedf8acd3bd110b48a95f4b9e546cd34f546effc05",
            ),
            (
                "microsoft-bing-image-creator.jpg",
                "aec9a34e-665f-4549-9c02-9d29337a5de6",
                "55cad5355b6eeeb25f84ea074f85c0ea55b82d63c27f43e3429bb75f28734ac7",
            ),
            (
                "../visible/microsoft/provider-original.png",
                "975290d4-9c2f-4359-ad5a-92c5b5b850a4",
                "0e76fb8aa0540b18fee4d63ac4b5fd63c7eee310474e896d21d27e668403afe7",
            ),
        ],
    )
    def test_real_microsoft_invismark_fixture(self, filename, watermark_id, sha256):
        path = SAMPLES_DIR / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256

        report = identify(path, check_visible=False)

        assert report.platform == "Microsoft (Copilot / Designer)"
        soft_binding = next(signal for signal in report.signals if signal.name == "soft_binding")
        assert "com.microsoft.invismark.1" in soft_binding.detail
        assert watermark_id in soft_binding.detail
        assert any(signal.name == "invismark" for signal in report.signals)


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


@pytest.mark.skipif(not _wm_available(), reason="detect extra not installed")
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
    """China TC260 AIGC labels are detected without guessing the product."""

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
        assert _vendor_of("OpenAI (ChatGPT / GPT Image / DALL·E / Sora)") == "OpenAI"
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

    def test_registered_vendors_normalize(self):
        # Regression: these registered C2PA vendors returned None, so their claims never
        # entered clash detection (a coverage hole). They now normalize to one origin.
        assert _vendor_of("Microsoft (Copilot / Designer)") == "Microsoft"
        assert _vendor_of("Copilot") == "Microsoft"
        assert _vendor_of("ByteDance Volcano Engine") == "ByteDance"
        assert _vendor_of("BytePlus (ByteDance)") == "ByteDance"
        assert _vendor_of("Dreamina/1.2") == "ByteDance"
        assert _vendor_of("Canva (Magic Media)") == "Canva"
        assert _vendor_of("Black Forest Labs (FLUX)") == "Black Forest Labs"
        assert _vendor_of("Eleven Labs Inc.") == "ElevenLabs"
        assert _vendor_of("Ideogram") == "Ideogram"

    def test_ideogram_fixture_attributed_end_to_end(self):
        # Committed synthetic manifest signed by a test CA with CN "Ideogram, Inc":
        # attribution must name the platform while signer trust stays untrusted.
        # Reachability against the fixture, not a synthesized status set.
        fixture = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance" / "ideogram-c2pa.png"
        if not fixture.exists():  # the sdist ships no data/
            pytest.skip("provenance fixture not present")
        report = identify(fixture, check_visible=False, check_invisible=False)
        assert report.ai_from_metadata is True
        assert report.platform == "Ideogram"
        assert any("Ideogram" in w for w in report.watermarks)

    def test_ideogram_issuer_attributed(self):
        # Corpus evidence 2026-08-08: four uploads signed "Ideogram, Inc" read as
        # unknown-signer C2PA with no platform. The issuer token is the org prefix.
        platforms = {v.issuer: v.platform for v in C2PA_AI_VENDORS}
        assert platforms[b"Ideogram"] == "Ideogram"
        assert "Ideogram" in C2PA_IDENTITY_AI_ORGS
        assert _issuers_in(b"...CN=Ideogram, Inc...trainedAlgorithmicMedia") == ["Ideogram"]

    def test_bytedance_issuers_keep_the_source_product(self):
        platforms = {
            vendor.issuer: vendor.platform
            for vendor in C2PA_AI_VENDORS
            if vendor.org.startswith("ByteDance") or vendor.org.startswith("BytePlus")
        }
        assert platforms[b"volcengine"] == "ByteDance Volcano Engine"
        assert platforms[b"Byteplus"] == "BytePlus (ByteDance)"
        assert platforms[b"Dreamina"] == "ByteDance Dreamina"
        assert ("dreamina", "ByteDance Dreamina") in C2PA_CLAIM_GENERATOR_PLATFORMS


class TestIntegrityClashesHelper:
    def test_tc260_manufacturer_comes_from_the_registered_producer(self):
        assert _tc260_manufacturer_of("001191440101MA9Y9T4H7A00001") == "Alibaba"
        assert _tc260_manufacturer_of("0011999999999999999999999") is None
        assert _tc260_manufacturer_of("") is None

    def test_two_ai_vendors_clash(self):
        clashes = _integrity_clashes({"c2pa": "OpenAI", "exif_generator": "Ideogram"}, None, camera_has_ai_marker=True)
        assert len(clashes) == 1
        assert "OpenAI" in clashes[0]
        assert "Ideogram" in clashes[0]

    def test_same_vendor_two_signals_no_clash(self):
        # C2PA Google + Google SynthID provenance is consistent, not a contradiction.
        assert _integrity_clashes({"c2pa": "Google", "synthid": "Google"}, None, camera_has_ai_marker=True) == []

    def test_multi_actor_manifest_no_clash(self):
        # A multi-actor C2PA manifest names a product + the engine it wraps in ONE
        # valid chain (Microsoft Designer on OpenAI, Microsoft on Google, Adobe over
        # a Gemini original). The C2PA issuer attribution and SynthID provenance share
        # the same manifest source, so the differing vendors must NOT read as a clash.
        for c2pa_vendor, synthid_vendor in (("Microsoft", "OpenAI"), ("Microsoft", "Google"), ("Adobe", "Google")):
            assert (
                _integrity_clashes({"c2pa": c2pa_vendor, "synthid": synthid_vendor}, None, camera_has_ai_marker=True)
                == []
            )

    def test_matching_tc260_manufacturer_no_clash(self):
        assert _integrity_clashes({"c2pa": "ByteDance", "aigc": "ByteDance"}, None, camera_has_ai_marker=True) == []

    def test_different_tc260_manufacturer_clashes(self):
        clashes = _integrity_clashes({"c2pa": "OpenAI", "aigc": "ByteDance"}, None, camera_has_ai_marker=True)
        assert len(clashes) == 1
        assert "Conflicting AI-origin" in clashes[0]

    def test_bytedance_c2pa_plus_foreign_generator_clashes(self):
        # Coverage win: a transplanted ByteDance C2PA manifest next to an independent
        # foreign generator stamp is a laundering tell that went undetected before
        # ByteDance was added to _vendor_of.
        clashes = _integrity_clashes({"c2pa": "ByteDance", "exif_generator": "OpenAI"}, None, camera_has_ai_marker=True)
        assert len(clashes) == 1
        assert "ByteDance" in clashes[0]
        assert "OpenAI" in clashes[0]

    def test_manifest_vendor_vs_independent_signal_clashes(self):
        # A vendor named only inside the manifest still clashes with a genuinely
        # independent stamp (here an EXIF/XMP generator tag) naming a third vendor.
        clashes = _integrity_clashes(
            {"c2pa": "Microsoft", "synthid": "Google", "exif_generator": "Ideogram"},
            None,
            camera_has_ai_marker=True,
        )
        assert len(clashes) == 1
        assert "Ideogram" in clashes[0]

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

    def test_pixel_generative_edit_same_manifest_no_clash(self):
        # A Google Pixel that BOTH captures and runs on-device generative AI
        # (Magic Editor / Pixel Studio) records the capture and the AI edit in
        # ONE C2PA manifest -- the AI vendor is named only from that same
        # manifest (c2pa / synthid), independent of nothing. That is a legitimate
        # edit chain, NOT a camera-vs-AI contradiction, so rule 2 must stay quiet.
        assert (
            _integrity_clashes(
                {"c2pa": "Google", "synthid": "Google"},
                "Google Pixel (camera, C2PA capture)",
                camera_has_ai_marker=True,
            )
            == []
        )

    def test_camera_plus_independent_ai_marker_still_clashes(self):
        # But a camera capture next to an AI marker from a genuinely INDEPENDENT
        # source (EXIF/XMP generator, TC260 AIGC, ...) is still a laundering tell.
        clashes = _integrity_clashes(
            {"c2pa": "Google", "aigc": "China AIGC (TC260)"},
            "Google Pixel (camera, C2PA capture)",
            camera_has_ai_marker=True,
        )
        assert any("Camera-capture" in c for c in clashes)


class TestIntegrityClashEndToEnd:
    def _c2pa_jpeg(self, tmp_path: Path, blob: bytes) -> Path:
        path = tmp_path / "img.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_two_generator_stamps_clash(self, tmp_path: Path):
        # An OpenAI C2PA manifest (AI source) on an image that ALSO carries a
        # Qwen TC260 AIGC label = two independent generator stamps naming
        # different manufacturers -> a laundering tell.
        path = self._c2pa_jpeg(
            tmp_path,
            b"OpenAI ... trainedAlgorithmicMedia ... "
            b'<TC260:AIGC>{"Label":"1","ContentProducer":"91440101MA9Y9T4H7A"}</TC260:AIGC>',
        )
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes
        assert any("Conflicting AI-origin" in c for c in r.integrity_clashes)

    def test_single_stamp_no_clash(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes == []

    def test_camera_device_plus_ai_marker_clash(self, tmp_path: Path):
        # Integrity-clash rule #2: a camera-capture C2PA device token (Pixel
        # Camera) coexisting with an independent AI-generation marker (a China
        # TC260 AIGC label) -- a genuine camera capture is not AI-generated, so
        # the provenance is inconsistent (a laundering / spoofing tell).
        path = self._c2pa_jpeg(
            tmp_path,
            b'Pixel Camera ... <TC260:AIGC>{"Label":"1","ContentProducer":"91110102MACQD9K640"}</TC260:AIGC>',
        )
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.platform == "Google Pixel (camera, C2PA capture)"
        assert any("Camera-capture C2PA credentials" in c and "AI-generation markers" in c for c in r.integrity_clashes)

    def test_pixel_generative_edit_no_clash(self, tmp_path: Path):
        # A real Google Pixel generative edit (Magic Editor / Pixel Studio) signs
        # ONE manifest carrying both the Pixel Camera capture and a Google
        # Generative AI edit (trainedAlgorithmicMedia + "Applied imperceptible
        # SynthID watermark"). The AI marker lives in the SAME manifest as the
        # device, so it is an edit chain, not a camera-vs-AI contradiction.
        path = self._c2pa_jpeg(
            tmp_path,
            b"Pixel Camera ... Created by Pixel Camera ... computationalCapture ... "
            b"Created by Google Generative AI ... trainedAlgorithmicMedia ... "
            b"Applied imperceptible SynthID watermark",
        )
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.is_ai_generated is True
        assert r.integrity_clashes == []

    def test_clash_serializes_to_json(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI ... trainedAlgorithmicMedia ... TC260:AIGC label")
        r = identify(path, check_visible=False, check_invisible=False)
        payload = json.loads(json.dumps(asdict(r), default=str))
        assert payload["integrity_clashes"] == r.integrity_clashes


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/fixtures/provenance not present")
@pytest.mark.parametrize("fixture", ["chatgpt-1.png", "firefly-1.png", "doubao-1.png", "grok-1.jpg", "mj-1.png"])
class TestRealSamplesHaveNoClash:
    """Every real single-origin fixture must report zero clashes (false-positive guard)."""

    def test_no_false_positive_clash(self, fixture: str):
        path = SAMPLES_DIR / fixture
        if not path.exists():
            pytest.skip(f"{fixture} not present")
        r = identify(path, check_visible=False, check_invisible=False)
        assert r.integrity_clashes == []


class TestSharedPixelDecode:
    """Every pixel detector in one report reads ONE decode of the source.

    ``identify`` used to decode the file three times -- the DWT-DCT detector, the
    TrustMark detector and the visible-mark stage each opened it. TrustMark still
    keeps its own PIL decode on purpose (cv2 and PIL disagree on EXIF orientation and
    on 16-bit PNG, so substituting one for the other is not behavior-preserving).
    """

    SAMPLE = SAMPLES_DIR / "chatgpt-1.png"

    def _count_source_decodes(self, **kwargs) -> int:
        from unittest.mock import patch

        from remove_ai_watermarks import image_io

        real = image_io.imread
        # Warm the bundled-asset caches first: alpha templates and sparkle captures
        # also go through imread, and they are one-time loads, not source decodes.
        identify(self.SAMPLE, check_visible=True, check_invisible=True)
        seen: list[str] = []

        def counted(path, *args, **kw):
            seen.append(str(path))
            return real(path, *args, **kw)

        with patch.object(image_io, "imread", counted):
            identify(self.SAMPLE, **kwargs)
        return sum(1 for s in seen if s == str(self.SAMPLE))

    @pytest.mark.skipif(not (SAMPLES_DIR / "chatgpt-1.png").exists(), reason="sample not present")
    def test_visible_and_invisible_share_one_decode(self):
        assert self._count_source_decodes(check_visible=True, check_invisible=True) == 1

    @pytest.mark.skipif(not (SAMPLES_DIR / "chatgpt-1.png").exists(), reason="sample not present")
    def test_metadata_only_report_never_decodes(self):
        assert self._count_source_decodes(check_visible=False, check_invisible=False) == 0

    @pytest.mark.skipif(not (SAMPLES_DIR / "chatgpt-1.png").exists(), reason="sample not present")
    def test_a_decode_failure_still_propagates_on_the_invisible_arm(self):
        """Load-bearing: ``has_invisible_target`` turns this exception into its
        documented fail-safe ``True``. Swallowing it would skip a diffusion scrub on a
        file that used to get one -- a watermark left on a paid removal."""
        from unittest.mock import patch

        from remove_ai_watermarks import identify as identify_mod
        from remove_ai_watermarks import image_io

        def boom(*args, **kwargs):
            raise OSError("decode exploded")

        with patch.object(image_io, "imread", boom):
            with pytest.raises(OSError, match="decode exploded"):
                identify(self.SAMPLE, check_visible=False, check_invisible=True)
            assert identify_mod.has_invisible_target(self.SAMPLE) is True

    @pytest.mark.skipif(not (SAMPLES_DIR / "chatgpt-1.png").exists(), reason="sample not present")
    def test_a_decode_failure_is_swallowed_on_the_visible_arm(self):
        """The visible arm's historical behavior: no cv2, no visible marks, and the
        metadata verdict is untouched."""
        from unittest.mock import patch

        from remove_ai_watermarks import image_io

        def boom(*args, **kwargs):
            raise OSError("decode exploded")

        with patch.object(image_io, "imread", boom):
            report = identify(self.SAMPLE, check_visible=True, check_invisible=False)
        assert not any(s.name.startswith("visible_") for s in report.signals)
        assert report.is_ai_generated is True  # the C2PA verdict survives the decode failure


class TestSynthIdProxyIsDecidedInTheVerdict:
    """The SynthID byte scan belongs to the verdict, not to extraction.

    Extraction has two implementations -- one reading a file, one reading a portable
    record -- so a rule that lives in only one of them is a rule the other silently
    lacks. This one did: 74 corpus images reported SynthID through ``identify`` and
    not through the record."""

    # A JUMBF-wrapped manifest from a SynthID-pairing signer on AI-generated content:
    # the exact shape `synthid_source`'s byte scan is gated on. Spliced into a real
    # JPEG as a well-formed APP11 segment, because a malformed one is skipped by the
    # record's structural walk and the test would compare two different inputs.
    MANIFEST = b"jumb c2pa Google LLC trainedAlgorithmicMedia"

    def _jpeg_with_manifest(self, path: Path) -> Path:
        import numpy as np
        from PIL import Image

        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path, "JPEG")
        data = path.read_bytes()
        segment = b"\xff\xeb" + (len(self.MANIFEST) + 2).to_bytes(2, "big") + self.MANIFEST
        path.write_bytes(data[:2] + segment + data[2:])
        return path

    def test_both_paths_infer_it_from_the_same_bytes(self, tmp_path: Path):
        from remove_ai_watermarks.identify import identify_metadata_record
        from remove_ai_watermarks.metadata_record import collect_metadata_record

        path = self._jpeg_with_manifest(tmp_path / "gemini.jpg")

        via_file = identify(path, check_visible=False, check_invisible=False)
        via_record = identify_metadata_record(collect_metadata_record(path), path=path)

        assert any("SynthID" in mark for mark in via_file.watermarks)
        assert via_record.watermarks == via_file.watermarks

    def test_it_needs_a_manifest_and_an_ai_source_type(self, tmp_path: Path):
        """The vendor name alone is not evidence: an ordinary photo mentioning
        "Google LLC" in EXIF must not acquire a SynthID verdict."""
        import numpy as np
        from PIL import Image

        path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path, "JPEG")
        data = path.read_bytes()
        note = b"Google LLC Pixel"
        path.write_bytes(data[:2] + b"\xff\xeb" + (len(note) + 2).to_bytes(2, "big") + note + data[2:])

        report = identify(path, check_visible=False, check_invisible=False)

        assert not any("SynthID" in mark for mark in report.watermarks)


class TestRegistryScansSkipTheCodedPixels:
    """The vendor registries match short raw substrings -- the shortest are four and
    five bytes. Over a megabyte of compressed pixel data such a sequence turns up by
    chance: `Bria` matched inside the entropy-coded scan of 4 of 14,707 corpus JPEGs,
    and that entry asserts AI, so a chance match can declare an image AI-generated.
    `c2pa_marker_in` already refuses a bare `c2pa` substring for the same reason."""

    def _jpeg(self, path: Path, *, in_segment: bytes = b"", in_scan: bytes = b"") -> Path:
        import numpy as np
        from PIL import Image

        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path, "JPEG")
        data = path.read_bytes()
        if in_segment:
            payload = b"jumb c2pa trainedAlgorithmicMedia " + in_segment
            data = data[:2] + b"\xff\xeb" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]
        if in_scan:
            # After SOS, i.e. inside the entropy-coded scan the walk skips.
            sos = data.index(b"\xff\xda")
            data = data[: sos + 16] + in_scan + data[sos + 16 :]
        path.write_bytes(data)
        return path

    def test_a_token_in_a_marker_segment_is_attributed(self, tmp_path: Path):
        from remove_ai_watermarks.identify import _issuers_in, _metadata_region
        from remove_ai_watermarks.metadata import scan_head

        path = self._jpeg(tmp_path / "signed.jpg", in_segment=b"Bria")

        assert _issuers_in(_metadata_region(scan_head(path))) == ["Bria Artificial Intelligence"]

    def test_a_token_in_the_coded_scan_is_not(self, tmp_path: Path):
        from remove_ai_watermarks.identify import _issuers_in, _metadata_region
        from remove_ai_watermarks.metadata import scan_head

        path = self._jpeg(tmp_path / "chance.jpg", in_segment=b"OpenAI", in_scan=b"Bria")
        region = _metadata_region(scan_head(path))

        assert _issuers_in(region) == ["OpenAI"]

    def test_a_container_that_does_not_parse_is_left_whole(self, tmp_path: Path):
        """Cutting a buffer the walk did not understand would drop real evidence to
        avoid a chance match, which is the wrong way round."""
        from remove_ai_watermarks.identify import _metadata_region

        blob = b"\xff\xd8" + b"not really a jpeg, no valid marker chain here" * 4

        assert _metadata_region(blob) == blob

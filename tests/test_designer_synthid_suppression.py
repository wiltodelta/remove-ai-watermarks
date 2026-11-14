"""Regression: a manifest that names its own watermark soft binding must not
also report a SynthID watermark from the generic vendor-token inference.

Microsoft Designer manifests sign as Microsoft, carry the InvisMark
``c2pa.watermarked`` action, and name their generation agent
"Azure OpenAI ImageGen". The OpenAI issuer token inside that agent name plus
the watermarked action used to satisfy the OpenAI SynthID-evidence rule,
double-counting one forensic mark as two pixel watermarks.
"""

from __future__ import annotations

from pathlib import Path

from remove_ai_watermarks._internal.c2pa import c2pa_info_from_manifest_store
from remove_ai_watermarks.identify import (
    evidence_from_metadata_record,
    identify_from_evidence,
    report_has_removal_hint,
)

DESIGNER_STORE = {
    "active_manifest": "designer",
    "manifests": {
        "designer": {
            "signature_info": {"issuer": "Microsoft Corporation", "common_name": "Microsoft Corporation"},
            "claim_generator_info": [{"name": "Microsoft Responsible AI Provenance", "version": "1.0"}],
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "softwareAgent": {"name": "Azure OpenAI ImageGen"},
                                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            },
                            {"action": "c2pa.watermarked"},
                        ]
                    },
                },
                {
                    "label": "c2pa.soft-binding",
                    "data": {
                        "alg": "com.microsoft.invismark.1",
                        "blocks": [{"value": "bf7a2993-cc1f-47e1-b1f0-cd8839aabb22"}],
                    },
                },
            ],
        }
    },
}


def test_named_watermark_soft_binding_suppresses_generic_synthid_evidence() -> None:
    info = c2pa_info_from_manifest_store(DESIGNER_STORE)
    assert info["ai_source_kind"] == "generated"
    assert info["soft_binding_algorithm"] == "com.microsoft.invismark.1"
    assert info.get("synthid_watermark") is None
    assert info.get("synthid_vendors") is None


def test_content_fingerprint_does_not_suppress_google_synthid_evidence() -> None:
    store = {
        "active_manifest": "google",
        "manifests": {
            "google": {
                "signature_info": {"issuer": "Google LLC"},
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                    ),
                                }
                            ]
                        },
                    },
                    {"label": "c2pa.soft-binding", "data": {"alg": "io.iscc.v0"}},
                ],
            }
        },
    }

    info = c2pa_info_from_manifest_store(store)

    assert info["soft_binding"] == "ISCC (content code)"
    assert info["synthid_vendors"] == ["Google LLC"]
    assert info["synthid_watermark"] == "present according to Google LLC provenance"


def test_vendor_agent_name_alone_is_not_the_vendors_provenance() -> None:
    """The identity-scoped inference must not fire on a service name either.

    Same manifest without the soft binding: the "Azure OpenAI ImageGen" agent
    is not an OpenAI signature or claim generator, so no OpenAI SynthID
    evidence may be derived from it.
    """
    store = {
        "active_manifest": "designer",
        "manifests": {
            "designer": {
                "signature_info": {"issuer": "Microsoft Corporation", "common_name": "Microsoft Corporation"},
                "claim_generator_info": [{"name": "Microsoft Responsible AI Provenance", "version": "1.0"}],
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "softwareAgent": {"name": "Azure OpenAI ImageGen"},
                                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                                },
                                {"action": "c2pa.watermarked"},
                            ]
                        },
                    }
                ],
            }
        },
    }
    info = c2pa_info_from_manifest_store(store)
    assert info.get("synthid_watermark") is None


def test_adobe_partner_model_parameter_digital_source_type_establishes_ai_without_false_synthid() -> None:
    """Adobe partner-model exports store digitalSourceType inside action.parameters.

    They also name the remote 3rd-party model (e.g. Gemini Flash) inside
    parameters. The nested com.adobe.digitalSourceType must establish AI-generated
    provenance even under a generic Adobe signer (such as Adobe Photoshop), while
    the parameter-only Gemini string must not fabricate Google SynthID provenance.
    """
    firefly_partner_store = {
        "active_manifest": "urn:uuid:partner-firefly",
        "manifests": {
            "urn:uuid:partner-firefly": {
                "signature_info": {"issuer": "Adobe Inc.", "common_name": "Adobe Firefly C2PA"},
                "claim_generator": "Adobe_Firefly",
                "claim_generator_info": [{"name": "Adobe Firefly", "org.contentauth.c2pa_rs": "0.90.4"}],
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.opened",
                                    "parameters": {
                                        "com.adobe.details": "Gemini Flash",
                                        "com.adobe.digitalSourceType": (
                                            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                        ),
                                        "com.adobe.modelId": "gemini-flash",
                                        "com.adobe.modelVersion": "nano-banana-3",
                                        "com.adobe.type": "remoteProvider.3rdParty",
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        },
    }
    info = c2pa_info_from_manifest_store(firefly_partner_store)
    assert info["source_type"] == "trainedAlgorithmicMedia (AI-generated)"
    assert info["ai_source_kind"] == "generated"
    assert info.get("synthid_watermark") is None
    assert info.get("synthid_vendors") is None

    # Under a generic Adobe Creative Cloud signer (no "Adobe_Firefly" identity),
    # com.adobe.digitalSourceType is the ONLY AI marker in the manifest.
    photoshop_partner_store = {
        "active_manifest": "urn:uuid:partner-ps",
        "manifests": {
            "urn:uuid:partner-ps": {
                "signature_info": {"issuer": "Adobe Inc.", "common_name": "Adobe Photoshop"},
                "claim_generator": "Adobe_Photoshop",
                "claim_generator_info": [{"name": "Adobe Photoshop"}],
                "assertions": firefly_partner_store["manifests"]["urn:uuid:partner-firefly"]["assertions"],
            }
        },
    }
    evidence = evidence_from_metadata_record({"c2pa_store": photoshop_partner_store}, path=Path("partner.png"))
    report = identify_from_evidence(evidence)
    assert report.is_ai_generated is True
    assert report.ai_from_metadata is True
    assert report.ai_source_kind == "generated"
    assert not any("SynthID" in w for w in report.watermarks)
    assert report_has_removal_hint(report, evidence.c2pa_info) is True


_GOOGLE_HEADERS: dict[str, object] = {
    "signature_info": {"issuer": "Google LLC", "common_name": "Google Media Processing Services"},
    "claim_generator_info": [{"name": "Google C2PA Core Generator Library"}],
}


def _google_gen_manifest(*, ingredient_manifest: str | None = None) -> dict[str, object]:
    manifest: dict[str, object] = {
        **_GOOGLE_HEADERS,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": (
                                "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                            ),
                            "description": "Created by Google Generative AI.",
                        },
                        {
                            "action": "c2pa.edited",
                            "digitalSourceType": (
                                "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                            ),
                            "description": "Applied imperceptible SynthID watermark.",
                        },
                    ]
                },
            }
        ],
    }
    if ingredient_manifest is not None:
        manifest["ingredients"] = [{"active_manifest": ingredient_manifest}]
    return manifest


def test_firefly_ingredient_edited_by_google_generative_ai_preserves_synthid_verdict() -> None:
    """Multi-manifest Firefly ingredient edited by Google Generative AI keeps SynthID provenance."""
    store = {
        "active_manifest": "urn:c2pa:google-export",
        "manifests": {
            "urn:c2pa:firefly-base": {
                "signature_info": {"issuer": "Adobe Inc."},
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.placed",
                                    "softwareAgent": {"name": "Adobe Firefly"},
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                    ),
                                }
                            ]
                        },
                    }
                ],
            },
            "urn:c2pa:google-gen": _google_gen_manifest(ingredient_manifest="urn:c2pa:firefly-base"),
            "urn:c2pa:google-export": {
                **_GOOGLE_HEADERS,
                "ingredients": [{"active_manifest": "urn:c2pa:google-gen"}],
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {"action": "c2pa.opened"},
                                {
                                    "action": "c2pa.edited",
                                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/composite",
                                    "description": "Added visible watermark",
                                },
                                {"action": "c2pa.converted", "description": "Converted to .png"},
                            ]
                        },
                    }
                ],
            },
        },
    }
    evidence = evidence_from_metadata_record({"c2pa_store": store}, path=Path("chained.png"))
    report = identify_from_evidence(evidence)
    assert report.is_ai_generated is True
    assert report.ai_from_metadata is True
    assert report.ai_source_kind == "generated"
    assert any("SynthID watermark (present according to Google LLC provenance)" in w for w in report.watermarks)
    assert report_has_removal_hint(report, evidence.c2pa_info) is True


def test_google_c2pa_chain_with_algorithmically_enhanced_transcode_preserves_generated_verdict() -> None:
    """A Google C2PA transcode wrapper (algorithmicallyEnhanced) keeps inner AI + SynthID provenance."""
    store = {
        "active_manifest": "urn:c2pa:google-transcode",
        "manifests": {
            "urn:c2pa:google-origin": _google_gen_manifest(),
            "urn:c2pa:google-transcode": {
                **_GOOGLE_HEADERS,
                "ingredients": [{"active_manifest": "urn:c2pa:google-origin"}],
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {"action": "c2pa.opened", "description": "Opened by Google"},
                                {
                                    "action": "c2pa.transcoded",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"
                                    ),
                                },
                            ]
                        },
                    }
                ],
            },
        },
    }
    evidence = evidence_from_metadata_record({"c2pa_store": store}, path=Path("reshared.png"))
    report = identify_from_evidence(evidence)
    assert report.is_ai_generated is True
    assert report.ai_from_metadata is True
    assert report.platform == "Google (Gemini / Imagen)"
    assert report.ai_source_kind == "generated"
    assert any("SynthID watermark (present according to Google LLC provenance)" in w for w in report.watermarks)
    assert report_has_removal_hint(report, evidence.c2pa_info) is True

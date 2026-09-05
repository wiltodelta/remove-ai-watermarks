"""Image provenance: identify where an image was made and what watermarks it carries.

Aggregates every locally-readable signal into a single :class:`ProvenanceReport`:

- **C2PA Content Credentials** (issuer, claim generator, digital source type) ->
  the signing platform (OpenAI, Google, Adobe, Microsoft).
- **IPTC ``digitalSourceType``** "Made with AI" marker (Meta, X, others).
- **PNG text / EXIF generation parameters** (Stable Diffusion, ComfyUI, InvokeAI).
- **SynthID provenance evidence** -- Google AI C2PA follows Google's all-media
  policy; current OpenAI C2PA explicitly declares a watermark action.
- **Registered visible marks** (optional; needs cv2/numpy, no GPU) through the
  shared watermark registry.

Hard limit: a stripped image (re-encoded, screenshotted, social-media upload)
loses all metadata, and the SynthID *pixel* watermark is not locally decodable
(proprietary decoder). Absence of signals is therefore reported as ``Unknown``,
never as "clean". See CLAUDE.md "SynthID detection is metadata-only".
"""

from __future__ import annotations

import base64
import itertools
import logging
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from remove_ai_watermarks._internal.c2pa import (
    c2pa_info_from_manifest_store,
    c2pa_info_has_invalid_credential,
    c2pa_info_has_invismark,
    c2pa_info_has_removal_hint,
    cbor_text_after,
    extract_c2pa_info,
    soft_binding_labels,
    soft_binding_registry_entries_in,
    synthid_evidence_vendors_in,
    synthid_verdict,
)
from remove_ai_watermarks._internal.constants import (
    C2PA_AI_TOOLS,
    C2PA_AI_VENDORS,
    C2PA_CLAIM_GENERATOR_PLATFORMS,
    C2PA_IDENTITY_AI_ORGS,
    C2PA_ISSUERS,
)
from remove_ai_watermarks._internal.schema import require_schema_version
from remove_ai_watermarks.metadata import (
    AI_METADATA_KEYS,
    AIGC_MARKERS,
    IPTC_AI_FIELD_MARKERS,
    IPTC_AI_MARKERS,
    aigc_label,
    aigc_label_from_metadata,
    c2pa_cloud_manifest_in,
    c2pa_marker_in,
    exif_generator,
    generator_from_metadata,
    get_ai_metadata,
    huggingface_job,
    iptc_ai_system,
    iptc_ai_system_in,
    samsung_genai,
    samsung_genai_in,
    scan_head,
    xai_signature,
    xai_signature_pair,
)
from remove_ai_watermarks.watermark_registry import GEMINI_SPARKLE_TRUST_CONF
from remove_ai_watermarks.watermark_registry import known_marks as _known_marks

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import MarkDetection

logger = logging.getLogger(__name__)

# Stable JSON contract for callers that pass a verdict between services. Bump this
# only for a breaking shape or semantic change; adding optional fields is compatible.
PROVENANCE_REPORT_SCHEMA_VERSION = 1

# How much of a non-PNG container to binary-scan for the C2PA issuer.
_SCAN_BYTES = 1024 * 1024

# Visible-sparkle confidence above which the signal is trusted as provenance.
# Shared with the removal arbitration (watermark_registry.GEMINI_SPARKLE_TRUST_CONF)
# so the provenance "is there a sparkle" verdict and the removal "take the sparkle"
# decision can never drift apart. Calibration showed that 0.5 separates Gemini-family
# sparkles from non-sparkle images and avoids
# false positives when the sparkle is the only signal (e.g. an OpenAI image scored
# 0.37 -- below threshold, correctly dropped).
_SPARKLE_THRESHOLD = GEMINI_SPARKLE_TRUST_CONF

# Issuer (C2PA signer) -> human-readable generating platform, derived from the
# single C2PA_AI_VENDORS registry. Ordered: when a manifest names several issuers
# (Microsoft Designer signs as "OpenAI, Microsoft"), the first match wins so the
# product, not the backend, is named -- the registry order encodes that priority.
# Signing authorities without an AI platform (e.g. Truepic) are skipped here.
_ISSUER_PLATFORM: tuple[tuple[str, str], ...] = tuple(
    (v.needle, v.platform) for v in C2PA_AI_VENDORS if v.platform is not None and v.needle is not None
)

# PNG-text / EXIF keys that indicate a local diffusion pipeline (vs. a hosted
# platform's C2PA). Subset of AI_METADATA_KEYS; excludes the C2PA/Software keys.
_LOCAL_GEN_KEYS = frozenset(
    AI_METADATA_KEYS & {"parameters", "prompt", "negative_prompt", "workflow", "comfyui", "invokeai_metadata", "dream"}
)

_STRIP_CAVEAT = (
    "Absence of metadata is not proof the image is clean: C2PA, EXIF, and PNG "
    "text chunks are stripped by re-encoding, screenshots, or social-media upload."
)
_SYNTHID_CAVEAT = (
    "SynthID presence comes from supported provenance here; the pixel watermark is not locally "
    "decoded (proprietary decoder). Confirm via the Gemini app or openai.com/verify."
)
_C2PA_UNTRUSTED_CAVEAT = (
    "The C2PA claim signature and asset binding validate, but no trust anchor list is configured here, "
    "so the signer identity was never checked against one; treat the named platform as a signed claim, "
    "not a verified identity."
)
_C2PA_EXPIRED_CAVEAT = (
    "The C2PA signing certificate has expired and no trusted timestamp establishes when the claim was "
    "signed. The binding to these bytes still holds; only the signing time is unproven."
)
_C2PA_UNVALIDATED_CAVEAT = (
    "The C2PA marker was parsed without cryptographic validation; treat its origin and watermark "
    "assertions as unverified claims."
)
_C2PA_INVALID_CAVEAT = (
    "The embedded C2PA claim no longer validates against this asset. Its origin and watermark assertions "
    "are retained only as removal hints, not as verified provenance."
)
_IPTC_ONLY_CAVEAT = "The IPTC 'Made with AI' tag flags AI provenance but does not identify the specific platform."
_CONTENT_SEAL_CAVEAT = (
    "Meta Muse Image outputs carry the invisible Content Seal pixel watermark, which has no "
    "local decoder; `invisible` removes it (auto when this tag is present, or `--vendor meta` "
    "on stripped files) and meta.ai/identification verifies it."
)
_INVISIBLE_WM_CAVEAT = (
    "The open invisible watermark is fragile: it does not survive JPEG re-encoding "
    "or resizing, so it confirms origin only on a pristine (un-re-encoded) file."
)
_HF_JOB_CAVEAT = (
    "The hf-job-id tag marks a Hugging Face-hosted job (commonly diffusion "
    "generation) but names neither the model nor the content type, so it is a "
    "medium-confidence signal, not proof the pixels are AI-generated."
)
_C2PA_CLOUD_CAVEAT = (
    "The embedded C2PA manifest is absent but an XMP provenance pointer to the "
    "vendor's cloud manifest store survives, so the Content Credentials remain "
    "recoverable server-side -- stripping the file no longer removes the provenance. "
    "It marks Content Credentials, not AI origin: the cloud manifest may describe a "
    "human edit, and reading it needs a network fetch this tool does not make."
)
_PIXEL_DETECTORS_MISSING_CAVEAT = (
    "The visible-mark detectors did NOT run: this install has no pixel dependencies, so "
    "a visible AI label may be present and unreported. Install "
    "'remove-ai-watermarks[visible]' and rerun before reading this scan as complete."
)
_PIXEL_DETECTORS_UNDECODABLE_CAVEAT = (
    "The visible-mark detectors did NOT run: the pixels could not be decoded, so a "
    "visible AI label may be present and unreported. This says nothing about the image "
    "itself -- only that this scan could not look at it."
)
_SOFT_BINDING_CAVEAT = (
    "Removing the embedded C2PA manifest does not break its soft binding: a named watermark may remain in the pixels "
    "or other media, while a content fingerprint may still be recomputed and re-link the asset to provenance."
)
_SAMSUNG_GENAI_CAVEAT = (
    "Samsung's genAIType marker shows a Galaxy AI editing tool (Generative Edit, "
    "Sketch to Image, ...) touched the image; it is an undocumented proprietary "
    "field, so it is a medium-confidence signal of AI editing, not proof the "
    "whole image is AI-generated."
)


@dataclass
class Signal:
    """A single provenance signal that was found (or affirmatively absent)."""

    name: str
    detail: str
    confidence: str  # "high" | "medium"


@dataclass(frozen=True)
class ProvenanceEvidence:
    """Extracted metadata evidence used by provenance detection.

    Extraction is intentionally separate from verdict logic so a caller can
    collect the file-backed evidence once and evaluate it without reopening the
    source. Pixel-backed visible and invisible watermark checks remain part of
    :func:`identify`.
    """

    path: Path
    c2pa_info: dict[str, Any]
    ai_metadata: dict[str, str]
    scan: bytes
    iptc_ai_system: str | None
    aigc_label: dict[str, str] | None
    exif_generator: str | None
    xai_signature: bool
    huggingface_job: str | None
    samsung_genai: int | None


def _external_metadata(value: Any) -> tuple[list[tuple[str, Any]], bytes]:
    """Index nested metadata and recover common encoded binary values in one pass."""
    pairs: list[tuple[str, Any]] = []
    parts: list[bytes] = []
    diagnostic_keys = {
        "artifacts",
        "birthtime",
        "color",
        "content_format",
        "dct",
        "ela",
        "error",
        "extension",
        "fft",
        "file",
        "filename",
        "full",
        "gradient",
        "kind",
        "mtime",
        "name",
        "noise",
        "path",
        "pixel",
        "provenance",
        "sha256",
        "signals",
        "size_bytes",
        "timing_ms",
    }

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            mapping = cast("dict[object, Any]", item)
            for key, nested in mapping.items():
                key_text = str(key)
                pairs.append((key_text, nested))
                if key_text.lower() in diagnostic_keys:
                    continue
                if isinstance(nested, str) and (key_text == "base64" or key_text.endswith("_base64")):
                    encoded = nested.split("...TRUNCATED", 1)[0]
                    try:
                        parts.append(base64.b64decode(encoded, validate=True))
                        continue
                    except (ValueError, TypeError):
                        pass
                visit(nested)
        elif isinstance(item, (list, tuple)):
            sequence = cast("list[Any] | tuple[Any, ...]", item)
            for nested in sequence:
                visit(nested)
        elif isinstance(item, bytes):
            parts.append(item)
        elif isinstance(item, str):
            if item.startswith("hex:"):
                try:
                    parts.append(bytes.fromhex(item[4:]))
                    return
                except ValueError:
                    pass
            parts.append(item.encode("utf-8", "replace"))
        elif item is not None:
            parts.append(str(item).encode("utf-8", "replace"))

    visit(value)
    return pairs, b"\n".join(parts)


def _external_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", "replace").strip()
    if not isinstance(value, str):
        return str(value).strip()
    if value.startswith("hex:"):
        try:
            return bytes.fromhex(value[4:]).decode("latin-1", "replace").strip()
        except ValueError:
            pass
    return value.strip()


def _external_exif_generator(pairs: list[tuple[str, Any]], scan: bytes) -> str | None:
    candidate_keys = {
        "software",
        "make",
        "artist",
        "imagedescription",
        "source",
        "title",
        "description",
        "creatortool",
        "usercomment",
    }
    candidates = [
        _external_text(value)
        for key, value in pairs
        if key.lower().removeprefix("info:") in candidate_keys and isinstance(value, (str, bytes))
    ]
    return generator_from_metadata(candidates, scan)


def _metadata_source_kind(info: dict[str, Any], scan: bytes) -> str | None:
    """Normalize the source type wherever it is carried: C2PA or IPTC/XMP.

    A composite marker contains ``TrainedAlgorithmicMedia`` as a substring, so it
    is removed before looking for a standalone full-generation marker. When a file
    genuinely carries both kinds, full generation wins.
    """
    structured = info.get("ai_source_kind")
    without_composites = scan.replace(b"compositeWithTrainedAlgorithmicMedia", b"").replace(b"compositeSynthetic", b"")
    generated = structured == "generated" or any(
        marker in without_composites for marker in (b"trainedAlgorithmicMedia", b"TrainedAlgorithmicMedia")
    )
    if generated:
        return "generated"
    if structured == "enhanced" or any(
        marker in scan for marker in (b"compositeWithTrainedAlgorithmicMedia", b"compositeSynthetic")
    ):
        return "enhanced"
    return None


def evidence_from_metadata_record(
    record: dict[str, Any], *, path: Path, c2pa_manifest_store: str | dict[str, Any] | None = None
) -> ProvenanceEvidence:
    """Normalize an externally collected metadata record into provenance evidence.

    Unversioned external records may contain arbitrary nested dictionaries and
    lists. Versioned native records accept only the source-derived fields emitted by
    ``collect_metadata_record``; other native record types and unknown schema
    versions are rejected. No source file is opened.
    """
    from remove_ai_watermarks.metadata_record import METADATA_RECORD_SCHEMA_VERSION, METADATA_RECORD_TYPE

    # Records produced by ``collect_metadata_record`` are a versioned transport
    # contract. Only their source-derived fields are evidence: the filename,
    # container label and schema bookkeeping describe the collector and must never
    # become detector input. Shape-detect the pre-versioned form as well so records
    # emitted by 0.26 remain safe and readable.
    record_type = record.get("record_type")
    if record_type not in (None, METADATA_RECORD_TYPE):
        raise ValueError(f"Unsupported metadata record type: {record_type!r}")
    if record_type == METADATA_RECORD_TYPE:
        require_schema_version(
            record.get("schema_version"),
            contract="provenance metadata",
            supported=(METADATA_RECORD_SCHEMA_VERSION,),
        )
        status = record.get("status")
        if status == "error":
            raise ValueError("Provenance metadata collection failed")
        if status != "complete":
            raise ValueError(f"Unsupported provenance metadata collection status: {status!r}")
    is_portable_record = record_type == METADATA_RECORD_TYPE or {
        "container",
        "metadata_base64",
        "tail_base64",
    }.issubset(record)
    evidence_record = (
        {key: record[key] for key in ("metadata_base64", "tail_base64", "pil", "exif") if key in record}
        if is_portable_record
        else record
    )
    pairs, scan = _external_metadata(evidence_record)
    store = c2pa_manifest_store
    if store is None:
        candidate = record.get("c2pa_store")
        store = (
            cast("dict[str, Any]", candidate)
            if isinstance(candidate, dict)
            else candidate
            if isinstance(candidate, str)
            else None
        )
    c2pa_info = c2pa_info_from_manifest_store(store) if store is not None else {}

    ai_metadata: dict[str, str] = {}
    pil_info = record.get("pil")
    pil_pairs = cast("dict[str, Any]", pil_info).items() if isinstance(pil_info, dict) else ()
    for key, value in pil_pairs:
        normalized_key = key.lower().removeprefix("info:")
        if normalized_key not in AI_METADATA_KEYS or isinstance(value, (dict, list, tuple)):
            continue
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        ai_metadata.setdefault(normalized_key, text[:200] + ("…" if len(text) > 200 else ""))
    for key, value in pairs:
        if key != "text" or not isinstance(value, str) or "\x00" not in value:
            continue
        metadata_key, metadata_value = value.split("\x00", 1)
        normalized_key = metadata_key.lower()
        if normalized_key in AI_METADATA_KEYS:
            ai_metadata.setdefault(
                normalized_key,
                metadata_value[:200] + ("…" if len(metadata_value) > 200 else ""),
            )
    for key in (
        "c2pa_manifest",
        "claim_generator",
        "c2pa_spec",
        "issuer",
        "source_type",
        "actions",
        "synthid_watermark",
        "soft_binding",
        "soft_binding_algorithm",
        "soft_binding_value",
    ):
        if key in c2pa_info:
            ai_metadata.setdefault(key, str(c2pa_info[key]))

    iptc_system = iptc_ai_system_in(scan)

    values_by_key: dict[str, str] = {}
    for key, value in pairs:
        if isinstance(value, (bytes, str)):
            values_by_key.setdefault(key.lower(), _external_text(value))
    description = values_by_key.get("imagedescription", "")
    artist = values_by_key.get("artist", "")
    xai = xai_signature_pair(description, artist)

    hf_job = next(
        (
            str(value).strip()
            for key, value in pairs
            if key.lower().removeprefix("info:") == "hf-job-id" and str(value).strip()
        ),
        None,
    )
    samsung = samsung_genai_in(scan)

    aigc_candidates = tuple(
        value for key, value in pairs if key.lower().removeprefix("info:") == "aigc" and isinstance(value, str)
    )
    aigc = aigc_label_from_metadata(scan, aigc_candidates)
    exif_gen = _external_exif_generator(pairs, scan)
    if aigc is not None:
        producer = aigc.get("ContentProducer", "")
        ai_metadata.setdefault(
            "aigc_label",
            f"China AIGC label (TC260){f'; producer {producer}' if producer else ''}",
        )
    if xai:
        ai_metadata.setdefault("xai_signature", "xAI/Grok EXIF signature (Artist UUID + Signature blob)")
    if iptc_system:
        ai_metadata.setdefault("ai_system", f"IPTC 2025.1 AI disclosure ({iptc_system})")
    if hf_job:
        ai_metadata.setdefault("huggingface_job", f"Hugging Face-hosted job ({hf_job})")
    if samsung is not None:
        ai_metadata.setdefault("samsung_genai", f"Samsung Galaxy AI editing marker (genAIType={samsung})")

    return ProvenanceEvidence(
        path=path,
        c2pa_info=c2pa_info,
        ai_metadata=ai_metadata,
        scan=scan,
        iptc_ai_system=iptc_system,
        aigc_label=aigc,
        exif_generator=exif_gen,
        xai_signature=xai,
        huggingface_job=hf_job,
        samsung_genai=samsung,
    )


@dataclass
class ProvenanceReport:
    """Aggregated provenance verdict for one image."""

    path: Path
    is_ai_generated: bool | None  # True / False is never asserted; None = unknown
    platform: str | None
    confidence: str  # "high" | "medium" | "none"
    # Coarse AI-origin kind from a C2PA or standalone IPTC/XMP digital-source-type,
    # so a caller can branch on full generation vs an AI-touched real photo:
    #   "generated" -- digitalSourceType trainedAlgorithmicMedia (fully AI).
    #   "enhanced"  -- compositeWithTrainedAlgorithmicMedia (real content with an
    #                  AI-composited region; scrub the AI region, keep the photo).
    #   None        -- no AI digital-source-type (verdict, if AI, came from another
    #                  signal: AIGC, local gen params, xAI, ...).
    ai_source_kind: str | None = None
    # True when the AI verdict rests on a metadata or embedded-invisible signal
    # (C2PA AI issuer / SynthID provenance, IPTC, AIGC, local gen params, EXIF/xAI, or
    # an open DWT-DCT decode) -- as opposed to a visible mark, provenance-only
    # TrustMark, or a weak medium-confidence hint (hf-job, Samsung genAIType). This
    # is the set of signals an invisible/diffusion scrub targets: a visible-only
    # or no-signal image has it False. Callers should gate on intent, not on the
    # confidence string.
    ai_from_metadata: bool = False
    watermarks: list[str] = field(default_factory=list[str])
    signals: list[Signal] = field(default_factory=list["Signal"])
    caveats: list[str] = field(default_factory=list[str])
    # Contradictions between independent provenance signals (e.g. two different
    # AI vendors both claiming the image, or camera-capture credentials next to
    # AI-generation markers), and credentials that failed validation. Non-empty means
    # the provenance is internally inconsistent -- a strong tell of spoofed,
    # transplanted, or laundered metadata. A failed credential sends
    # ``is_ai_generated`` to None and lands here instead, so a consumer that reads
    # only the verdict field turns a broken AI manifest into silence; see the
    # ``integrity_clashes`` note in docs/python-api.md.
    integrity_clashes: list[str] = field(default_factory=list[str])
    # Orthogonal C2PA checks. A valid content binding does not make an untrusted
    # signer identity trusted, and an expired credential does not by itself mean the
    # signed bytes changed. Fallback parsing reports each dimension as unknown;
    # None means no C2PA result exists.
    c2pa_validation: dict[str, Any] | None = None

    def to_dict(
        self,
        *,
        schema_version: int = PROVENANCE_REPORT_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Return the versioned, JSON-safe verdict contract.

        ``path`` is deliberately omitted. It is extraction context, not part of the
        verdict, and local filesystem paths should not cross a service boundary.
        Request an explicit schema for a long-lived transport consumer.
        """
        schema_version = require_schema_version(
            schema_version,
            contract="provenance report",
            supported=(1,),
        )
        return {
            "schema_version": schema_version,
            "is_ai_generated": self.is_ai_generated,
            "platform": self.platform,
            "confidence": self.confidence,
            "ai_source_kind": self.ai_source_kind,
            "ai_from_metadata": self.ai_from_metadata,
            "watermarks": list(self.watermarks),
            "signals": [
                {
                    "name": signal.name,
                    "detail": signal.detail,
                    "confidence": signal.confidence,
                }
                for signal in self.signals
            ],
            "caveats": list(self.caveats),
            "integrity_clashes": list(self.integrity_clashes),
            "c2pa_validation": self.c2pa_validation,
        }


def _c2pa_validation(info: dict[str, Any]) -> dict[str, Any] | None:
    source = info.get("c2pa_validation_source")
    if not isinstance(source, str):
        return None
    codes = info.get("c2pa_validation_codes")
    return {
        "source": source,
        "state": str(info.get("c2pa_validation_state", "unknown")),
        "integrity": str(info.get("c2pa_integrity", "unknown")),
        "signature": str(info.get("c2pa_signature", "unknown")),
        "signer_trust": str(info.get("c2pa_signer_trust", "unknown")),
        "signer_validity": str(info.get("c2pa_signer_validity", "unknown")),
        "codes": list(cast("list[object]", codes)) if isinstance(codes, list) else [],
    }


def _c2pa_credential_level(info: dict[str, Any]) -> str:
    """Return invalid, verified, or unverified for provenance attribution.

    ``verified`` means the reader tied this manifest to these bytes: the hard binding
    matched and the claim signature validated. Signer trust is deliberately NOT a
    condition -- no trust anchors ship, so gating on it made this branch unreachable.
    Read the trust-anchor paragraph in docs/module-internals.md before changing this.
    """
    if c2pa_info_has_invalid_credential(info):
        return "invalid"
    if info.get("c2pa_integrity") == "valid" and info.get("c2pa_signature") == "valid":
        return "verified"
    return "unverified"


def extract_provenance_evidence(image_path: Path) -> ProvenanceEvidence:
    """Read all file-backed metadata needed by provenance verdict logic once."""
    return ProvenanceEvidence(
        path=image_path,
        c2pa_info=extract_c2pa_info(image_path),
        ai_metadata=get_ai_metadata(image_path),
        scan=scan_head(image_path, _SCAN_BYTES),
        iptc_ai_system=iptc_ai_system(image_path),
        aigc_label=aigc_label(image_path),
        exif_generator=exif_generator(image_path),
        xai_signature=xai_signature(image_path),
        huggingface_job=huggingface_job(image_path),
        samsung_genai=samsung_genai(image_path),
    )


def _issuers_in(data: bytes) -> list[str]:
    """C2PA issuer names whose signature byte appears in ``data`` (binary scan)."""
    return sorted({name for sig, name in C2PA_ISSUERS.items() if sig in data})


def _ai_tools_in(data: bytes) -> list[str]:
    """Known C2PA AI-tool / generator names appearing in ``data`` (binary scan).

    PNG has a structured claim_generator; for JPEG/WebP/AVIF/HEIF/JXL the
    generator lives in a JUMBF/EXIF/XMP blob the PNG parser can't reach, so a
    byte scan recovers the same attribution (e.g. "Imagen", "DALL-E").
    """
    return sorted({name for sig, name in C2PA_AI_TOOLS.items() if sig in data})


# Distinctive C2PA device/camera tokens (cert CN, cert org, or claim-generator
# substrings) scanned in the manifest bytes -> platform. This is more reliable
# than mapping an issuer name (which also matches incidental mentions: a
# timestamp authority like "Truepic" in a Leica chain, an XMP-toolkit "Adobe"
# string in a Nikon file, or "Google" in a Pixel camera's cert -- all verified
# on real samples), and more robust than parsing the claim generator (which
# lives under varying CBOR keys, e.g. `claim_generator` vs `claim_generator_info`,
# and is absent on the Pixel sample where only the cert CN "Pixel Camera"
# identifies it). Camera C2PA marks CAPTURE authenticity, not AI, so these never
# assert is_ai on their own (the verdict still comes from the digital-source-type:
# the Pixel sample carries `computationalCapture`, not `trainedAlgorithmicMedia`).
# Only tokens verified against a real signed file are listed (Leica, Nikon,
# Sony, Truepic, Google Pixel); add Canon/Bria as real samples are captured.
# Samsung Galaxy is an AI-capable editing device, not a pure-capture camera, so
# it lives in `_SIGNER_C2PA_PLATFORM` below (it must not feed the camera clash).
_DEVICE_C2PA_PLATFORM: tuple[tuple[bytes, str], ...] = (
    (b"lc_c2pa", "Leica (camera, C2PA capture)"),
    (b"Leica Camera", "Leica (camera, C2PA capture)"),
    (b"NIKON", "Nikon (camera, C2PA capture)"),
    (b"Pixel Camera", "Google Pixel (camera, C2PA capture)"),
    # Sony uses its own ``sony.*`` C2PA assertion namespace (sony.sig / sony.cert);
    # match that, NOT bare "Sony" (which is an EXIF Make on countless photos).
    # Verified on a real Sony-signed file (Sony PXW-Z300, signer "Sony Corporation").
    (b"sony.sig", "Sony (camera, C2PA capture)"),
    (b"sony.cert", "Sony (camera, C2PA capture)"),
    # "Truepic_Lens" (from the Lens SDK claim generator), NOT bare "Truepic" --
    # Truepic is a C2PA signing authority whose name appears in the trust chain
    # of unrelated manifests (e.g. OpenAI), so the bare token mis-attributes.
    (b"Truepic_Lens", "Truepic Lens (verified capture)"),
)


def _metadata_region(head: bytes) -> bytes:
    """The part of the scan buffer that can hold metadata, with the coded pixels cut out.

    The vendor registries are matched as raw substrings, and the shortest tokens are
    four and five bytes (``Bria``, ``Adobe``, ``Canva``). Over a megabyte of compressed
    pixel data a four-byte sequence appears by chance about once in three thousand
    images -- measured: ``Bria`` matched inside the entropy-coded scan of 4 of 14,707
    corpus JPEGs, in none of which the manifest names Bria. That is not a cosmetic
    mislabel, because the Bria entry carries ``asserts_ai``: a chance match can declare
    an image AI-generated.

    ``c2pa_marker_in`` already refuses a bare ``c2pa`` substring for the same reason.
    This is the same defence for the registries: they see the container's metadata and
    not its pixels.

    JPEG keeps the marker segments before the entropy-coded scan, PNG every chunk but
    ``IDAT``, and both keep the trailer past the end marker. Anything ``scan_head``
    APPENDED past the window is metadata by construction (late chunks, boxes, decoder
    text), so it is always kept and never walked -- walking it is what produced 11 MB
    records and a phantom AIGC signal in the record collector.

    Trimming happens only when the container actually parses: a JPEG whose marker walk
    reaches the coded scan, a PNG whose chunk walk reaches ``IDAT``. Anything else --
    a malformed container, a synthetic blob, a format with no walker here -- is
    returned whole. Cutting a buffer this function did not understand would drop real
    evidence to avoid a chance match, which is the wrong way round.
    """
    raw, appended = head[:_SCAN_BYTES], head[_SCAN_BYTES:]
    if raw[:2] == b"\xff\xd8":
        index, size = 2, len(raw)
        while index + 1 < size:
            if raw[index] != 0xFF:
                return head  # not a marker boundary: the walk is lost, keep everything
            marker = raw[index + 1]
            if marker in (0xDA, 0xD9):  # SOS / EOI: the coded scan follows
                end = raw.rfind(b"\xff\xd9")
                return raw[:index] + (raw[end + 2 :] if end >= index else b"") + appended
            if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                index += 2
                continue
            if index + 4 > size:
                break
            length = int.from_bytes(raw[index + 2 : index + 4], "big")
            if length < 2 or index + 2 + length > size:
                break
            index += 2 + length
        return head  # ran out of buffer before the scan: nothing was skipped anyway
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        out = bytearray()
        position, size, saw_idat = 8, len(raw), False
        while position + 8 <= size:
            (length,) = struct.unpack(">I", raw[position : position + 4])
            chunk_type = raw[position + 4 : position + 8]
            start = position + 8
            if chunk_type == b"IDAT":
                saw_idat = True
            else:
                out += chunk_type + raw[start : start + min(length, size - start)]
            position = start + length + 4
            if chunk_type == b"IEND":
                out += raw[position:]
                break
        return bytes(out) + appended if saw_idat else head
    return head


def _first_token_match(head: bytes, table: tuple[tuple[bytes, str], ...]) -> str | None:
    """First platform in ``table`` whose token appears in ``head``, else None.

    Table order is priority: the more specific token must be listed first.
    """
    for token, platform in table:
        if token in head:
            return platform
    return None


def _device_platform(head: bytes) -> str | None:
    """Map a distinctive C2PA device/camera token in the manifest bytes to a platform."""
    return _first_token_match(head, _DEVICE_C2PA_PLATFORM)


# C2PA signers that are an editing app or AI-capable device rather than a
# verified-capture camera. Unlike `_DEVICE_C2PA_PLATFORM`, these do NOT feed the
# camera-vs-AI integrity clash (rule 2 in `_integrity_clashes`): a Galaxy phone
# legitimately stamps BOTH its device credentials AND a `trainedAlgorithmicMedia`
# source type on a Generative-Edit image, so treating it as a "genuine camera
# capture" would false-flag every Galaxy AI edit. They only resolve the platform
# label; the AI verdict still comes from the digital-source-type / genAIType.
# Tokens verified against real signed files (2026-05-29):
#   Samsung Galaxy -- cert org on Galaxy S23 FE / S24 / S25 C2PA JPEGs/PNGs
#     (distinct from the EXIF "SM-xxxx" model string on ordinary Samsung photos).
#   com.asus.gallery -- ASUS Gallery claim_generator (a C2PA-signed edit, no AI
#     source type or genAIType on the samples, so it never asserts is_ai).
_SIGNER_C2PA_PLATFORM: tuple[tuple[bytes, str], ...] = (
    (b"Samsung Galaxy", "Samsung Galaxy (C2PA)"),
    (b"com.asus.gallery", "ASUS Gallery (C2PA signer)"),
)


def _signer_platform(head: bytes) -> str | None:
    """Map a C2PA editing-app / AI-capable-device signer token to a platform."""
    return _first_token_match(head, _SIGNER_C2PA_PLATFORM)


def _attribute_platform(issuers: list[str], *, is_ai: bool = True) -> str | None:
    """Map a set of C2PA issuer names to a human-readable generating platform.

    A specific AI-generator platform (Adobe Firefly, OpenAI, ...) is named only
    when the content is actually AI (``is_ai``, i.e. digital-source-type
    ``trainedAlgorithmicMedia``). Otherwise an issuer-name byte match is likely
    incidental -- e.g. an "Adobe XMP" toolkit string in a Canon/Sony camera
    capture, or a "Google" cert org -- so we fall back to a neutral signer label
    rather than mislabel a camera photo as "Adobe Firefly". Real Firefly/OpenAI/
    Google AI output carries the AI source-type, so it is unaffected. ``is_ai``
    defaults True so the issuer->platform mapping can still be unit-tested in
    isolation; ``identify`` passes the file's actual ``c2pa_is_ai``.
    """
    joined = " ".join(issuers)
    if is_ai:
        for needle, platform in _ISSUER_PLATFORM:
            if needle in joined:
                return platform
    if issuers:  # e.g. Truepic alone -- a signing authority, not a generator
        return f"C2PA signer: {', '.join(issuers)} (no known AI generator named)"
    return None


def _claim_generator_platform(generator: str | None) -> str | None:
    """Resolve a distinctive C2PA claim generator to its user-facing product."""
    if not generator:
        return None
    lowered = generator.lower()
    return next((platform for token, platform in C2PA_CLAIM_GENERATOR_PLATFORMS if token in lowered), None)


# Coarse origin-vendor normalization for integrity-clash detection. Two signals
# that resolve to the SAME key are consistent (a C2PA "Google (Gemini)" issuer
# and Google SynthID provenance, or Adobe Firefly + its Adobe TrustMark soft
# binding); two DIFFERENT keys from independent generator stamps are a
# contradiction (a C2PA OpenAI manifest on an image whose EXIF says "Ideogram
# AI"). Substring match on the lowercased platform/detail string; first hit wins,
# so order specific tokens before brand umbrellas where they overlap.
_AI_VENDOR_TOKENS: tuple[tuple[str, str], ...] = (
    ("gpt-image", "OpenAI"),
    ("dall", "OpenAI"),
    ("sora", "OpenAI"),
    ("openai", "OpenAI"),
    ("gemini", "Google"),
    ("imagen", "Google"),
    ("nano banana", "Google"),
    ("google", "Google"),
    ("firefly", "Adobe"),
    ("adobe", "Adobe"),
    ("copilot", "Microsoft"),
    ("bing", "Microsoft"),
    ("designer", "Microsoft"),
    ("microsoft", "Microsoft"),
    ("stability", "Stability AI"),
    ("stable diffusion", "Stability AI"),
    ("sdxl", "Stability AI"),
    ("ideogram", "Ideogram"),
    ("grok", "xAI"),
    ("aurora", "xAI"),
    ("xai", "xAI"),
    # ByteDance family (all its brands normalize to one origin, mirroring constants.py):
    # without these a transplanted ByteDance C2PA manifest next to an independent
    # conflicting stamp went undetected by the clash check.
    ("bytedance", "ByteDance"),
    ("doubao", "ByteDance"),
    ("jimeng", "ByteDance"),
    ("dreamina", "ByteDance"),
    ("volcengine", "ByteDance"),
    ("volcano engine", "ByteDance"),
    ("canva", "Canva"),
    ("elevenlabs", "ElevenLabs"),
    ("eleven labs", "ElevenLabs"),
    ("black forest", "Black Forest Labs"),
    ("fal-ai", "fal.ai"),
    ("bria", "Bria"),
    ("apple photos clean up", "Apple"),
    ("luma ai", "Luma AI"),
    ("lumalabs", "Luma AI"),
)


def _vendor_of(text: str | None) -> str | None:
    """Normalize a platform/generator string to a coarse origin-vendor key, or None."""
    if not text:
        return None
    low = text.lower()
    for token, vendor in _AI_VENDOR_TOKENS:
        if token in low:
            return vendor
    return None


# Clash-detection provenance sources. Rule 1 (below) flags two AI vendors only
# when they come from *independent* signals. The C2PA issuer attribution and the
# SynthID evidence are NOT independent -- both are read from the same C2PA
# manifest -- so they share one source. A multi-actor manifest (a product wrapping
# another vendor's engine, e.g. Microsoft+OpenAI or Microsoft+Google; or an edit
# chain like Adobe over a Gemini original) legitimately names several vendors in
# one valid chain and must not read as spoofing. Families not listed here are each
# their own independent source (EXIF/XMP generator, IPTC AISystemUsed, AIGC, ...).
# The single C2PA-manifest source shared by the issuer attribution and the SynthID
# evidence (both read from the same embedded manifest). Rule 2 keys off it too:
# the camera device label is read from this manifest, so an AI marker is a clash
# only when its source differs from this (i.e. it is genuinely independent).
_C2PA_MANIFEST_SOURCE = "c2pa_manifest"
_CLASH_SOURCE: dict[str, str] = {"c2pa": _C2PA_MANIFEST_SOURCE, "synthid": _C2PA_MANIFEST_SOURCE}

# The generic China TC260 AIGC vendor label -- a COUNTRY-LEVEL regulatory "this is AI"
# stamp any Chinese generator applies to its own output, naming no specific vendor.
_GENERIC_AIGC_VENDOR = "China AIGC (TC260)"
# Vendors that apply the TC260 label to their OWN output. When one is co-attributed with
# the generic AIGC label, the label is that vendor's own stamp (not an independent
# competing origin), so the clash check attributes the AIGC label to it -- else a legit
# ByteDance/Doubao image (C2PA "ByteDance" + its own TC260 label) would false-clash once
# ByteDance normalizes via _vendor_of. Chinese generators only (Canva/BFL/ElevenLabs,
# also added to _vendor_of, are NOT TC260 appliers).
_TC260_VENDORS: frozenset[str] = frozenset({"ByteDance"})


def _integrity_clashes(
    ai_vendors: dict[str, str], camera_label: str | None, *, camera_has_ai_marker: bool
) -> list[str]:
    """Surface contradictions between independent provenance signals.

    Args:
        ai_vendors: family name -> normalized AI-origin vendor, one entry per
            generator-stamped signal (C2PA issuer when the source is AI, SynthID
            provenance, EXIF/XMP generator tag, IPTC AISystemUsed, xAI, AIGC label).
        camera_label: a camera/verified-capture C2PA device platform, if one was
            identified (Pixel, Leica, Sony, Nikon, Truepic), else None.
        camera_has_ai_marker: True when an AI-generation stamp coexists with the
            camera credentials.

    Returns:
        Human-readable clash descriptions; empty when the signals agree.
    """
    clashes: list[str] = []

    # Rule 1: two genuinely INDEPENDENT signals naming different AI vendors. Two
    # families clash only when they belong to different provenance sources (see
    # _CLASH_SOURCE) AND name different vendors -- so multiple vendors named within
    # one C2PA manifest (C2PA issuer + SynthID provenance) do not flag.
    # The generic TC260 AIGC label is a Chinese regulatory "this is AI" stamp. When a
    # Chinese TC260-applying vendor (ByteDance) is ALSO attributed, the label is that
    # vendor's own stamp on its own output, so attribute it to that vendor -- a legit
    # Doubao image carries BOTH a ByteDance C2PA manifest and its own TC260 label and
    # must not clash. Against a NON-TC260 vendor (OpenAI, Google, ...) the label stays
    # generic and still clashes as a laundering tell (a foreign-vendor image carrying a
    # Chinese TC260 label names two different origins).
    if ai_vendors.get("aigc") == _GENERIC_AIGC_VENDOR:
        own = next((v for f, v in ai_vendors.items() if f != "aigc" and v in _TC260_VENDORS), None)
        if own:
            ai_vendors = {**ai_vendors, "aigc": own}  # copy co-located with the relabel

    source = {fam: _CLASH_SOURCE.get(fam, fam) for fam in ai_vendors}
    independent_conflict = any(
        source[a] != source[b] and ai_vendors[a] != ai_vendors[b] for a, b in itertools.combinations(ai_vendors, 2)
    )
    if independent_conflict:
        by_vendor: dict[str, list[str]] = {}
        for family, vendor in ai_vendors.items():
            by_vendor.setdefault(vendor, []).append(family)
        parts = [f"{vendor} (via {', '.join(sorted(fams))})" for vendor, fams in sorted(by_vendor.items())]
        clashes.append(
            "Conflicting AI-origin attributions from independent signals: "
            + " vs ".join(parts)
            + " -- one provenance set was likely spoofed, transplanted, or laundered."
        )

    # Rule 2: a camera-capture C2PA device next to an AI-generation marker. Only
    # an AI marker from a source INDEPENDENT of the camera's own C2PA manifest is
    # a contradiction. A device that both captures and runs on-device generative
    # AI (Google Pixel Magic Editor / Pixel Studio) records the capture and the
    # AI edit in ONE manifest, so the AI vendor is named only from that same
    # manifest (C2PA issuer + SynthID provenance) -- a legitimate edit chain, not a
    # spoof. An EXIF/XMP generator, IPTC field, TC260 AIGC label, or second
    # manifest naming AI on a camera capture is the real laundering tell.
    independent_ai_marker = any(grp != _C2PA_MANIFEST_SOURCE for grp in source.values())
    if camera_label and camera_has_ai_marker and independent_ai_marker:
        vendors = ", ".join(sorted(set(ai_vendors.values()))) or "present"
        clashes.append(
            f"Camera-capture C2PA credentials ({camera_label}) coexist with AI-generation markers "
            f"({vendors}) -- a genuine camera capture is not AI-generated, so the provenance is inconsistent."
        )

    return clashes


def _visible_sparkle(image_path: Path, *, image: NDArray[Any] | None = None) -> float | None:
    """Visible Gemini-sparkle confidence in [0, 1], or None if unavailable.

    Optional: needs cv2/numpy (no GPU). The cv2 work lives in gemini_engine so
    this module stays dependency-light; returns None if cv2 or the engine
    assets are missing, or the image can't be read. ``image`` is a pre-decoded
    BGR array shared across the visible-mark detectors (see ``identify``) so the
    file is not decoded once per detector.
    """
    try:
        from remove_ai_watermarks.gemini_engine import detect_sparkle_confidence
    except Exception as exc:  # cv2/engine assets missing
        logger.debug("visible-sparkle detector unavailable: %s", exc)
        return None
    return detect_sparkle_confidence(image_path, image=image)


# Visible text marks (registry keys) -> human-readable platform, mirroring the
# Gemini-sparkle phrasing. These are the stripped-metadata visual fallback for
# the China-served ByteDance generators (normally also caught by the TC260 AIGC
# metadata label); the per-engine detection thresholds live in the registry.
# Text mark -> the platform sentence this report prints when that mark is the strongest
# evidence, DERIVED from the registry rows so registering a mark is one edit. It was a
# hand-maintained copy, and that class of copy is how LiblibAI ended up registered but
# missing from the pill veto. Insertion order is the registry's, which is what fixes the
# scan order below. The Gemini sparkle and the capture-less pill carry no platform of
# their own (`KnownMark.platform is None`) and are excluded here: the sparkle has its
# own higher-confidence `_visible_sparkle` path.
#
# Safe at module scope: `watermark_registry` is already imported above for
# GEMINI_SPARKLE_TRUST_CONF, and it is deliberately cv2-free at import time.
_VISIBLE_MARK_PLATFORM: dict[str, str] = {
    mark.key: mark.platform for mark in _known_marks() if mark.platform is not None
}


def _visible_text_marks(image_path: Path, *, image: NDArray[Any] | None = None) -> list[MarkDetection]:
    """Detected visible text marks (registry ``MarkDetection`` list).

    The Gemini sparkle keeps its own ``_visible_sparkle`` path (file-level
    confidence); the text marks reuse the registry detectors, which apply
    each engine's calibrated NCC threshold via ``MarkDetection.detected``.
    Optional: needs cv2/numpy; returns ``[]`` if the engines/assets are missing
    or the image can't be read. ``image`` is a pre-decoded BGR array shared
    across the visible-mark detectors (see ``identify``) so the file is not
    decoded once per detector.
    """
    try:
        from remove_ai_watermarks.image_io import imread
        from remove_ai_watermarks.watermark_registry import get_mark
    except Exception as exc:  # cv2/engine assets missing
        logger.debug("visible-mark detectors unavailable: %s", exc)
        return []
    if image is None:
        image = imread(image_path)
    if image is None:
        return []
    detections: list[MarkDetection] = []
    for key in _VISIBLE_MARK_PLATFORM:
        try:
            det = get_mark(key).detect(image)
        except Exception as exc:  # one engine failing must not break identify
            logger.debug("visible-mark %s detector failed: %s", key, exc)
            continue
        if det.detected:
            detections.append(det)
    return detections


def _invisible_watermark(image_path: Path, decode: _SharedDecode) -> str | None:
    """Open invisible-watermark scheme name (SD/SDXL/FLUX) or None.

    Optional: needs the torch-free DWT-DCT decoder (extra ``detect``). Returns
    None if it is not installed or no known watermark decodes.
    """
    from remove_ai_watermarks.invisible_watermark import detect_invisible_watermark, is_available

    if not is_available():
        return None
    # `decode.get()` re-raises a decode failure exactly as the old unguarded
    # `imread` inside the detector did -- `has_invisible_target` needs that to reach
    # its fail-safe rather than silently reporting "no signal".
    return detect_invisible_watermark(image_path, image=decode.get())


def _trustmark(image_path: Path) -> str | None:
    """Adobe TrustMark scheme name or None.

    Optional: needs the ``trustmark`` decoder (extra ``trustmark``). Returns None
    if it is not installed or no TrustMark watermark decodes.
    """
    from remove_ai_watermarks.trustmark_detector import detect_trustmark

    return detect_trustmark(image_path)


class _SharedDecode:
    """One decode of the source pixels, shared by every detector in a single report.

    ``identify`` used to decode the file three times. This holder unifies TWO of
    them -- the DWT-DCT detector and the visible-mark stage, whose own docstring
    already promised a single shared array. TrustMark keeps its own Pillow decode
    on purpose and is NOT served from here: cv2 and Pillow disagree on EXIF
    orientation and on 16-bit PNG, so feeding it this array would change what it
    decodes. An install carrying the optional ``trustmark`` extra therefore still
    pays two decodes, not one.

    Two accessors, because the two arms need OPPOSITE failure handling:

    * :meth:`get_or_none` swallows a decode failure and logs it. That is the visible
      arm's historical behavior -- no cv2, no visible marks, verdict unchanged.
    * :meth:`get` RE-RAISES it. The invisible arm never caught a decode error, and
      ``has_invisible_target`` converts that exception into its documented fail-safe
      ``True``. Swallowing it here would silently skip a diffusion scrub on a file
      that used to get one -- leaving a watermark on a paid removal.

    Per-CALL only: constructed inside ``_identify_from_evidence`` and discarded with
    it, so an in-place rewrite between calls can never be answered from a stale array.
    """

    __slots__ = ("_done", "_error", "_image", "_path")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done = False
        self._image: NDArray[Any] | None = None
        self._error: Exception | None = None

    def _decode(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            from remove_ai_watermarks.image_io import imread

            self._image = imread(self._path)
        except Exception as exc:  # cv2 missing / unreadable container
            self._error = exc

    def get(self) -> NDArray[Any] | None:
        """The decoded array; re-raises a decode failure, None only on a clean miss."""
        self._decode()
        if self._error is not None:
            raise self._error
        return self._image

    def get_or_none(self) -> NDArray[Any] | None:
        """The decoded array, or None when it could not be decoded at all."""
        self._decode()
        if self._error is not None:
            logger.debug("visible-mark decode unavailable: %s", self._error)
            return None
        return self._image

    @property
    def error(self) -> Exception | None:
        """The decode failure, so a caller can tell an absent extra from a bad file."""
        return self._error


def _collect_visible_signals(
    image_path: Path,
    signals: list[Signal],
    watermarks: list[str],
    platform: str | None,
    decode: _SharedDecode,
    caveats: list[str],
) -> str | None:
    """Append every trusted visible-mark signal and return platform.

    All visible detectors share the one decoded BGR array held by ``decode`` (which
    the invisible detectors have usually already paid for). A decode failure
    preserves the detectors' historical fallback/no-op behavior.
    """
    image = decode.get_or_none()
    if image is None:
        # The no-op is deliberate and the verdict is unchanged, but the REPORT used to
        # stay silent, so a scan that could not look reads exactly like a scan that
        # looked and found nothing. An absent extra and an unreadable file both land
        # here and need different fixes, so the caveat names which one happened rather
        # than sending the user to reinstall a working install.
        from remove_ai_watermarks.optional_deps import pixels_available

        missing_stack = isinstance(decode.error, ImportError) and not pixels_available()
        caveats.append(_PIXEL_DETECTORS_MISSING_CAVEAT if missing_stack else _PIXEL_DETECTORS_UNDECODABLE_CAVEAT)
        return platform

    sparkle_conf = _visible_sparkle(image_path, image=image)
    if sparkle_conf is not None and sparkle_conf >= _SPARKLE_THRESHOLD:
        signals.append(Signal("visible_sparkle", f"NCC confidence {sparkle_conf:.2f}", "medium"))
        watermarks.append(f"Google Gemini visible watermark (sparkle; confidence {sparkle_conf:.2f})")
        if platform is None:
            platform = "Google Gemini family (visible sparkle detected)"

    for detection in _visible_text_marks(image_path, image=image):
        signals.append(Signal(f"visible_{detection.key}", f"NCC confidence {detection.confidence:.2f}", "medium"))
        watermarks.append(f"Visible {detection.label} (confidence {detection.confidence:.2f})")
        if platform is None:
            platform = _VISIBLE_MARK_PLATFORM[detection.key]
    return platform


def _identify_from_evidence(
    evidence: ProvenanceEvidence,
    *,
    image_path: Path | None = None,
    check_visible: bool = False,
    check_invisible: bool = False,
) -> ProvenanceReport:
    """Build a provenance verdict from extracted evidence.

    ``image_path`` is supplied only by :func:`identify` for optional pixel
    detectors. Metadata-only callers leave it unset and never reopen the source.
    """
    if (check_visible or check_invisible) and image_path is None:
        raise ValueError("Pixel-backed checks require image_path")
    pixel_path = image_path
    # One decode for every pixel detector in this report. Built here, per call, so it
    # dies with the report -- an in-place rewrite between two calls cannot be answered
    # from a stale array. Lazy inside, so a metadata-only report never decodes at all.
    decode = _SharedDecode(pixel_path) if pixel_path is not None else _SharedDecode(evidence.path)

    info = evidence.c2pa_info
    meta = evidence.ai_metadata
    head = evidence.scan

    signals: list[Signal] = []
    watermarks: list[str] = []
    caveats: list[str] = []
    # One normalized origin vendor per generator-stamped signal, for integrity-
    # clash detection (see _integrity_clashes). Visible sparkle and the open
    # invisible watermark are deliberately excluded: the former is a fuzzy visual
    # score, the latter can be a by-product of our own SDXL removal pass, so
    # neither is a trustworthy "the generator stamped its identity" claim.
    ai_vendor_claims: dict[str, str] = {}
    # The vendor registries match short raw substrings, so they read the container's
    # metadata rather than its pixels -- see `_metadata_region`. Every other check
    # below keeps the full buffer: their markers are long and distinctive.
    region = _metadata_region(head)
    camera_label = _device_platform(region)
    signer_label = _signer_platform(region)

    # ── C2PA Content Credentials ────────────────────────────────────
    has_c2pa = bool(info) or c2pa_marker_in(head)
    c2pa_validation = _c2pa_validation(info)
    c2pa_level = _c2pa_credential_level(info)
    c2pa_usable = c2pa_level != "invalid"
    # The reader already named which failures moved a dimension; re-deriving that here
    # by substring made the displayed reason a second, looser rule than the verdict.
    failed_c2pa_codes = [str(code) for code in cast("list[object]", info.get("c2pa_failed_codes", []))]
    issuers = [info["issuer"]] if info.get("issuer") else _issuers_in(region)
    # Full AI generation (trainedAlgorithmicMedia) vs an AI-enhanced real photo
    # (compositeWithTrainedAlgorithmicMedia). The structured kind is parsed once in
    # _internal.c2pa._structured_manifest_fields (covers PNG + any container the c2pa-python
    # reader handles); fall back to a raw head scan for the non-PNG raw-blob path
    # where extract_c2pa_info returns {}. Full generation wins when both appear.
    source_kind = _metadata_source_kind(info, head)
    # An identity-AI issuer (a pure-generator brand like Dreamina) asserts AI even
    # without a digitalSourceType -- some ByteDance/Dreamina manifests ship no
    # trainedAlgorithmicMedia, so the registered generator name is the only signal.
    # Restricted to the ``asserts_ai`` vendors (distinctive brand strings), so it
    # does not reopen the incidental-mention problem the common-word issuers have.
    issuer_blob = " ".join(issuers)
    c2pa_identity_ai = has_c2pa and (
        bool(info.get("c2pa_identity_ai")) or any(org in issuer_blob for org in C2PA_IDENTITY_AI_ORGS)
    )
    c2pa_claims_ai = source_kind is not None or c2pa_identity_ai
    c2pa_is_ai = c2pa_usable and c2pa_claims_ai
    # Generator string (for the signal detail): structured for PNG, CBOR-scanned
    # for other containers. Best-effort -- some manifests key it as
    # `claim_generator_info` (Pixel), so this can be None even when a device is
    # identified by `_device_platform`.
    generator = (
        info.get("claim_generator")
        or cbor_text_after(head, b"claim_generator")
        or (", ".join(tools) if (tools := _ai_tools_in(region)) else None)
    )
    # Platform: a distinctive device/camera token in the manifest wins (it is the
    # signer/producer), then an editing-app/AI-device signer (Samsung Galaxy,
    # ASUS Gallery), with the issuer byte-scan only as fallback. The issuer scan
    # alone mis-attributed real samples (Leica->Truepic timestamp authority,
    # Nikon->Adobe namespace, Pixel->Google Gemini) -- the token scans fix that.
    platform = (
        (
            camera_label
            or signer_label
            or (_claim_generator_platform(generator) if c2pa_is_ai else None)
            or (_claim_generator_platform(str(info.get("ai_tool"))) if c2pa_is_ai and info.get("ai_tool") else None)
            or _attribute_platform(issuers, is_ai=c2pa_is_ai)
        )
        if has_c2pa and c2pa_usable
        else None
    )
    if has_c2pa:
        detail = ", ".join(filter(None, [", ".join(issuers), generator, info.get("source_type")]))
        if c2pa_level == "invalid":
            suffix = f"; {', '.join(failed_c2pa_codes)}" if failed_c2pa_codes else ""
            signals.append(Signal("c2pa", f"C2PA manifest present, credential integrity invalid{suffix}", "medium"))
            watermarks.append("C2PA Content Credentials (invalid asset binding or signature)")
            caveats.append(_C2PA_INVALID_CAVEAT)
        else:
            signals.append(
                Signal("c2pa", detail or "C2PA manifest present", "high" if c2pa_level == "verified" else "medium")
            )
            watermarks.append(f"C2PA Content Credentials ({', '.join(issuers) or 'unknown signer'})")
            if c2pa_level == "verified":
                # A missing trust bundle is not a signer that failed against one.
                if info.get("c2pa_signer_trust") != "trusted":
                    caveats.append(_C2PA_UNTRUSTED_CAVEAT)
                if info.get("c2pa_signer_validity") == "expired":
                    caveats.append(_C2PA_EXPIRED_CAVEAT)
            else:
                caveats.append(_C2PA_UNVALIDATED_CAVEAT)
        # Record the AI-origin vendor for clash detection only when the source is
        # actually AI -- classify the issuer attribution / generator, NOT the
        # resolved `platform` (which may be a camera device token whose label,
        # e.g. "Google Pixel", would mis-normalize to an AI vendor).
        if c2pa_is_ai and (
            v := (
                _vendor_of(_attribute_platform(issuers, is_ai=True))
                or _vendor_of(generator)
                or _vendor_of(str(info.get("ai_tool", "")))
            )
        ):
            ai_vendor_claims["c2pa"] = v

    # ── C2PA cloud-manifest reference (Durable Content Credentials) ─
    # An XMP dcterms:provenance pointer to a vendor manifest store survives even
    # when the embedded manifest is stripped, so the credentials stay recoverable
    # server-side (C2PA 2.4). Provenance only -- it does NOT assert AI (the cloud
    # manifest may describe a human edit), so it is excluded from ai_from_metadata
    # and the clash vendors. Skip when an embedded manifest already attributed it.
    if not has_c2pa and (cloud_vendor := c2pa_cloud_manifest_in(head)):
        signals.append(Signal("c2pa_cloud", f"cloud manifest store: {cloud_vendor}", "medium"))
        watermarks.append(
            f"C2PA Durable Content Credentials (cloud manifest at {cloud_vendor}; embedded manifest absent)"
        )
        caveats.append(_C2PA_CLOUD_CAVEAT)
        if platform is None:
            platform = f"C2PA signer: {cloud_vendor} (cloud manifest)"

    # ── SynthID provenance evidence ─────────────────────────────────
    # Structured first (the PNG caBX parser and the manifest store both fill
    # `synthid_watermark`), then the byte scan for the containers that keep the
    # manifest where no parser reaches it.
    #
    # The scan lives HERE, in the verdict, and not in extraction, for the same reason
    # `soft_binding` below does: extraction has two implementations -- one reading a
    # file, one reading a portable record -- and a rule that lives in only one of them
    # is a rule the other silently lacks. It did: 74 corpus images reported SynthID
    # through `identify` and not through the record, because `get_ai_metadata`'s own
    # fallback has no counterpart on the record side. `get_ai_metadata` keeps its copy
    # for its own callers; the verdict no longer depends on which extractor ran.
    synthid = meta.get("synthid_watermark")
    # The literal byte checks mirror `metadata.synthid_source` exactly rather than
    # reusing the derived `has_c2pa` / `source_kind` above, which are broader:
    # the file path's answer must not move.
    trained_source = b"trainedAlgorithmicMedia" in head or b"TrainedAlgorithmicMedia" in head
    # Same suppression as every other inference site: bytes that name their own
    # watermark soft-binding carries that vendor's mark, and the generic
    # vendor-token inference must not add a second, differently-attributed mark
    # (Microsoft Designer: "Azure OpenAI ImageGen" agent + the InvisMark action
    # read as "SynthID per OpenAI"). Fingerprints do not suppress independent
    # SynthID evidence; an unknown algorithm stays fail-safe as a possible mark.
    soft_binding_algorithm = meta.get("soft_binding_algorithm") or info.get("soft_binding_algorithm")
    soft_binding_scan = region
    if soft_binding_algorithm:
        soft_binding_scan += b"\n" + str(soft_binding_algorithm).encode("utf-8", "replace")
    soft_binding_entries = soft_binding_registry_entries_in(soft_binding_scan)
    soft_binding_vendors = soft_binding_labels(soft_binding_entries)
    soft_binding_watermarks = soft_binding_labels(soft_binding_entries, kind="watermark")
    soft_binding_fingerprints = soft_binding_labels(soft_binding_entries, kind="fingerprint")
    soft_binding_blocks_synthid = bool(soft_binding_watermarks) or bool(
        soft_binding_algorithm and not soft_binding_entries
    )
    if (
        not synthid
        and trained_source
        and c2pa_marker_in(head)
        and not soft_binding_blocks_synthid
        and (vendors := synthid_evidence_vendors_in(region))
    ):
        synthid = synthid_verdict(", ".join(vendors))
    if synthid:
        watermarks.append(
            f"SynthID watermark ({synthid})"
            if c2pa_usable
            else f"SynthID watermark claimed by invalid C2PA credentials ({synthid})"
        )
        caveats.append(_SYNTHID_CAVEAT)
        if c2pa_usable and (v := _vendor_of(synthid)):
            ai_vendor_claims["synthid"] = v

    # ── C2PA soft-binding: a registered watermark or content fingerprint ─
    # Present in the manifest even when the binding cannot be resolved locally.
    soft_binding = meta.get("soft_binding") or (", ".join(soft_binding_vendors) if soft_binding_vendors else None)
    if soft_binding:
        soft_binding_value = meta.get("soft_binding_value") or info.get("soft_binding_value")
        soft_binding_details = "; ".join(
            str(value) for value in (soft_binding, soft_binding_algorithm, soft_binding_value) if value
        )
        signal_prefix = (
            "C2PA content fingerprint"
            if soft_binding_fingerprints and not soft_binding_watermarks
            else "C2PA soft binding"
        )
        signals.append(
            Signal(
                "soft_binding",
                f"{signal_prefix}: {soft_binding_details}",
                "high" if c2pa_level == "verified" else "medium",
            )
        )
        if c2pa_info_has_invismark(info):
            # Keep the generic soft-binding row for schema-1 compatibility, and add
            # the pixel-removal target as its own stable signal for clients that need
            # to select diffusion without parsing human-readable detail.
            signals.append(
                Signal(
                    "invismark",
                    f"Microsoft InvisMark pixel watermark: {soft_binding_details}",
                    "high" if c2pa_level == "verified" else "medium",
                )
            )
        if soft_binding_watermarks:
            watermark_details = (
                soft_binding_details if not soft_binding_fingerprints else ", ".join(soft_binding_watermarks)
            )
            watermarks.append(f"Forensic watermark soft binding ({watermark_details})")
        caveats.append(_SOFT_BINDING_CAVEAT)

    # ── IPTC "Made with AI" (Meta etc.), only meaningful without C2PA ─
    iptc = any(m in head for m in IPTC_AI_MARKERS)
    standalone_iptc = iptc and not has_c2pa
    if standalone_iptc:
        signals.append(Signal("iptc", "digitalSourceType (Made with AI)", "high"))
        watermarks.append("IPTC digitalSourceType (Made with AI)")
        # Muse Image stamps every output with the invisible Content Seal, and this
        # tag is the only provenance such a file carries - the same measured bet
        # the strength router makes (vendor_for_strength -> "meta"). Emit the seal
        # as its own stable signal, the way InvisMark is additive over
        # soft_binding, so clients select pixel removal from the signal list
        # instead of parsing caveats. It is an attribution, not a decode: no
        # public Content Seal decoder exists, hence "medium".
        signals.append(
            Signal(
                "content_seal",
                "Meta Muse Content Seal pixel watermark (attributed by the standalone AI digital-source tag)",
                "medium",
            )
        )
        watermarks.append("Invisible Content Seal watermark (Meta Muse attribution)")
        caveats.append(_IPTC_ONLY_CAVEAT)
        caveats.append(_CONTENT_SEAL_CAVEAT)
        if platform is None:
            # Apple Photos Clean Up (Apple Intelligence object removal) marks
            # the edit with photoshop:Credit / IPTC "Apple Photos Clean Up"
            # next to compositeWithTrainedAlgorithmicMedia. It was detected but
            # previously never attributed.
            if b"Apple Photos Clean Up" in head:
                platform = "Apple Photos (Clean Up AI edit)"
            else:
                # The platform line follows the same measured bet the seal signal
                # and the strength router make: Muse Image is the tag writer whose
                # outputs this profile targets, so a hedged Muse attribution is
                # more useful than "platform not specified" while the panel below
                # already prices the Content Seal removal. The hedge stays in the
                # wording - it names the attribution basis, not a detection.
                platform = "Meta Muse Image (attributed by the standalone AI digital-source tag)"

    # ── IPTC 2025.1 AI-disclosure fields (Iptc4xmpExt:AISystemUsed etc.) ─
    iptc_ai = any(m in head for m in IPTC_AI_FIELD_MARKERS)
    if iptc_ai:
        system = evidence.iptc_ai_system
        named = bool(system) and system != "fields present"
        signals.append(
            Signal("iptc_ai_system", f"IPTC AI disclosure ({system})" if named else "IPTC AI disclosure fields", "high")
        )
        watermarks.append(f"IPTC 2025.1 AI disclosure ({system})" if named else "IPTC 2025.1 AI disclosure fields")
        if platform is None and named:
            platform = f"{system} (IPTC AISystemUsed)"
        if named and (v := _vendor_of(system)):
            ai_vendor_claims["iptc_ai_system"] = v

    # ── China TC260 AIGC label (Doubao and other China-served gens) ──
    # Fire on either the namespaced byte marker (``TC260:AIGC`` / the TC260 ns
    # URL, present in XMP and as a laundering tell even when the JSON payload is
    # truncated) OR the parsed label, which additionally catches the raw-JSON
    # PNG ``AIGC`` tEXt chunk that carries no namespaced marker at all.
    aigc_data = evidence.aigc_label
    aigc = aigc_data is not None or any(m in head for m in AIGC_MARKERS)
    if aigc:
        producer = (aigc_data or {}).get("ContentProducer", "")
        signals.append(Signal("aigc", f"TC260 AIGC label{f' (producer {producer})' if producer else ''}", "high"))
        watermarks.append("China AIGC label (TC260 standard)")
        if platform is None:
            platform = "China AIGC-labeled generator (TC260; e.g. Doubao)"
        ai_vendor_claims["aigc"] = _GENERIC_AIGC_VENDOR

    # ── Local diffusion parameters (Stable Diffusion / ComfyUI) ──────
    local_keys = sorted(k for k in meta if k.lower() in _LOCAL_GEN_KEYS)
    if local_keys:
        signals.append(Signal("gen_params", f"embedded keys: {', '.join(local_keys)}", "high"))
        watermarks.append("Embedded generation parameters (Stable Diffusion / ComfyUI)")
        if platform is None:
            platform = "Stable Diffusion / local pipeline (Automatic1111, ComfyUI, InvokeAI)"

    # ── EXIF Software / XMP CreatorTool / PNG-text generator (cross-format) ─
    # Catches a generator tag (incl. inside AVIF/HEIF/JXL and PNG text chunks)
    # when there is no C2PA.
    if generator_tag := evidence.exif_generator:
        signals.append(Signal("exif_generator", f"Embedded generator tag: {generator_tag}", "high"))
        watermarks.append(f"Embedded generator tag: {generator_tag}")
        if platform is None:
            platform = f"{generator_tag} (embedded generator tag)"
        if v := _vendor_of(generator_tag):
            ai_vendor_claims["exif_generator"] = v

    # ── xAI / Grok EXIF signature scheme (no C2PA/SynthID/IPTC) ──────
    # Grok's only provenance signal: EXIF ImageDescription "Signature: <base64>"
    # + a UUID Artist. Distinct from exif_generator (which matches generator
    # tokens); verified stable across 3 generations. See CLAUDE.md.
    if evidence.xai_signature:
        signals.append(Signal("xai_signature", "EXIF Signature blob + UUID Artist", "high"))
        watermarks.append("xAI/Grok EXIF signature")
        if platform is None:
            platform = "xAI (Grok / Aurora)"
        ai_vendor_claims["xai"] = "xAI"

    # ── Hugging Face-hosted job marker (hf-job-id PNG text chunk) ─────
    # Marks the hosting job, not a model -- medium confidence (commonly diffusion
    # output). Like the visible sparkle, it lifts an otherwise-Unknown verdict to
    # a tentative AI, but never overrides a high-confidence metadata signal.
    hf_job = evidence.huggingface_job
    if hf_job:
        signals.append(Signal("hf_job", f"Hugging Face job {hf_job}", "medium"))
        watermarks.append("Hugging Face-hosted job (hf-job-id)")
        caveats.append(_HF_JOB_CAVEAT)
        if platform is None:
            platform = "Hugging Face-hosted job (model not identified)"

    # ── Samsung Galaxy AI editing marker (genAIType) ─────────────────
    # Galaxy AI tools stamp a proprietary genAIType in PhotoEditor_Re_Edit_Data.
    # Medium confidence: it co-occurs with the C2PA trainedAlgorithmicMedia type
    # on Galaxy files that record one, and is the SOLE AI marker on a Galaxy S24
    # sample that omits the source type -- so it lifts an otherwise-Unknown
    # verdict, but the field is undocumented, so it never overrides a high-
    # confidence signal. The platform is usually already "Samsung Galaxy" via the
    # signer-token scan; the fallback covers a future file without the cert org.
    samsung_genai_type = evidence.samsung_genai
    if samsung_genai_type is not None:
        signals.append(Signal("samsung_genai", f"Samsung genAIType={samsung_genai_type}", "medium"))
        watermarks.append("Samsung Galaxy AI editing marker (genAIType)")
        caveats.append(_SAMSUNG_GENAI_CAVEAT)
        if platform is None:
            platform = "Samsung Galaxy (Galaxy AI editing)"

    # ── Open invisible watermark (SD / SDXL / FLUX, dwtDct) ──────────
    # Public decoder, no key -- a definitive embedded signal on pristine files.
    if check_invisible and pixel_path is not None and (scheme := _invisible_watermark(pixel_path, decode)) is not None:
        signals.append(Signal("invisible_watermark", scheme, "high"))
        watermarks.append(f"Open invisible watermark: {scheme}")
        caveats.append(_INVISIBLE_WM_CAVEAT)
        if platform is None:
            platform = f"{scheme} (open DWT-DCT watermark)"

    # ── Adobe TrustMark invisible watermark (open decoder, no key) ───
    # The watermark behind Adobe Durable Content Credentials. Decoded locally,
    # but it binds provenance for human-authored content too, so it enriches the
    # watermark inventory without by itself asserting AI origin.
    if check_invisible and pixel_path is not None and (tm_scheme := _trustmark(pixel_path)) is not None:
        signals.append(Signal("trustmark", tm_scheme, "high"))
        watermarks.append(f"Adobe TrustMark invisible watermark ({tm_scheme})")
        if platform is None:
            platform = "Adobe (TrustMark / Content Credentials)"

    # ── Verdict so far (metadata + embedded watermark) ──────────────
    invisible_wm = any(s.name == "invisible_watermark" for s in signals)
    exif_gen = any(s.name == "exif_generator" for s in signals)
    xai_sig = any(s.name == "xai_signature" for s in signals)
    ai_from_metadata = bool(
        (has_c2pa and c2pa_usable and (c2pa_is_ai or synthid))
        or standalone_iptc
        or iptc_ai
        or aigc
        or local_keys
        or invisible_wm
        or exif_gen
        or xai_sig
    )
    high_ai_from_metadata = bool(
        (has_c2pa and c2pa_level == "verified" and (c2pa_is_ai or synthid))
        or standalone_iptc
        or iptc_ai
        or aigc
        or local_keys
        or invisible_wm
        or exif_gen
        or xai_sig
    )

    if check_visible and pixel_path is not None:
        platform = _collect_visible_signals(pixel_path, signals, watermarks, platform, decode, caveats)

    visible_only = any(s.name.startswith("visible_") for s in signals) and not ai_from_metadata
    hf_only = bool(hf_job) and not ai_from_metadata
    samsung_only = samsung_genai_type is not None and not ai_from_metadata

    if ai_from_metadata:
        is_ai: bool | None = True
        confidence = "high" if high_ai_from_metadata else "medium"
    elif visible_only or hf_only or samsung_only:
        is_ai = True
        confidence = "medium"
    else:
        is_ai = None
        confidence = "none"

    # ── Integrity clashes: contradictions between independent signals ─
    clashes = _integrity_clashes(
        ai_vendor_claims,
        camera_label if c2pa_usable else None,
        camera_has_ai_marker=bool(ai_vendor_claims),
    )
    if c2pa_level == "invalid":
        clashes.insert(
            0,
            "C2PA credentials failed integrity validation"
            + (f": {', '.join(failed_c2pa_codes)}" if failed_c2pa_codes else "."),
        )

    caveats.append(_STRIP_CAVEAT)
    # De-duplicate while preserving order.
    caveats = list(dict.fromkeys(caveats))

    return ProvenanceReport(
        path=evidence.path,
        is_ai_generated=is_ai,
        platform=platform,
        confidence=confidence,
        # Meaningful for the same digitalSourceType whether carried by C2PA or a
        # standalone IPTC/XMP label. Other AI signals leave it None.
        ai_source_kind=(source_kind if (is_ai and ((has_c2pa and c2pa_usable) or standalone_iptc)) else None),
        ai_from_metadata=ai_from_metadata,
        watermarks=watermarks,
        signals=signals,
        caveats=caveats,
        integrity_clashes=clashes,
        c2pa_validation=c2pa_validation,
    )


def identify_from_evidence(
    evidence: ProvenanceEvidence,
    *,
    image_path: Path | None = None,
    check_visible: bool = False,
    check_invisible: bool = False,
) -> ProvenanceReport:
    """Build a provenance verdict from already-extracted evidence.

    Metadata-only by default -- the source is never reopened. Pass ``image_path`` with
    ``check_visible`` / ``check_invisible`` to add the pixel-backed detectors on top of
    the SAME evidence, which is how a caller that asks the file two provenance questions
    (which vendor is confirmed, and is there an invisible target) pays for the metadata
    extraction once.
    """
    return _identify_from_evidence(
        evidence,
        image_path=image_path,
        check_visible=check_visible,
        check_invisible=check_invisible,
    )


def identify_metadata_record(record: dict[str, Any], *, path: Path) -> ProvenanceReport:
    """Build a metadata-only verdict from a portable metadata record.

    This is the service-integration entry point: the source file is never opened,
    and callers receive the same verdict as the explicit
    ``evidence_from_metadata_record`` / ``identify_from_evidence`` sequence.
    """
    return identify_from_evidence(evidence_from_metadata_record(record, path=path))


def identify(
    image_path: Path,
    *,
    check_visible: bool = True,
    check_invisible: bool = True,
) -> ProvenanceReport:
    """Identify an image's origin platform and watermark inventory.

    Args:
        image_path: Path to the image (PNG, JPEG, WebP, or ISOBMFF container).
        check_visible: Also run the registered visible-mark detectors through cv2.
            Set False for a metadata-only, dependency-light scan.
        check_invisible: Also decode optional open invisible watermarks
            (SD/SDXL/FLUX). No-op when the decoder extra is not installed.

    File-backed metadata extraction runs first. The extracted evidence is then
    evaluated independently, followed by the optional pixel-backed visible and
    invisible watermark checks.

    Returns:
        A :class:`ProvenanceReport`. ``is_ai_generated`` is True when any AI
        signal is found and None (unknown) when none is found. It is never
        asserted False because stripped metadata leaves no local proof of a
        clean origin.
    """
    evidence = extract_provenance_evidence(image_path)
    return _identify_from_evidence(
        evidence,
        image_path=image_path,
        check_visible=check_visible,
        check_invisible=check_invisible,
    )


def has_invisible_target(image_path: Path) -> bool:
    """True when a locally-detectable invisible/metadata AI signal is present.

    The decision gate for the diffusion scrub (``invisible`` / ``all`` / ``batch``):
    regenerating pixels removes an AI-specific invisible watermark (SynthID,
    open DWT-DCT) but degrades a real photo, so it must not run when there is
    nothing to remove. It runs the same evidence pipeline as :func:`identify`
    with visible checks disabled and invisible checks enabled, so a visible mark
    is handled by the separate visible pass and is NOT a diffusion target. Returns
    True for ``report.ai_from_metadata`` (C2PA AI issuer / SynthID provenance,
    IPTC, AIGC, local gen params, EXIF/xAI, or open DWT-DCT), and also when an
    invalid C2PA claim retains an AI or watermark removal hint. TrustMark alone
    does not trigger the scrub because it also protects human-authored work and
    therefore is not an AI signal by itself.

    IMPORTANT -- this cannot prove a pixel SynthID is absent: SynthID is detectable
    only through its C2PA proxy, so a metadata-stripped AI image reads as no signal
    here. A False therefore means "no locally-detectable invisible target", not
    "clean". Callers must NOT present a skip as a finished clean result.

    Fail-safe: any error resolves to True so the removal still runs -- leaving a
    watermark on a paid removal is worse than over-regenerating a clean image.
    """
    try:
        evidence = extract_provenance_evidence(image_path)
        report = _identify_from_evidence(
            evidence,
            image_path=image_path,
            check_visible=False,
            check_invisible=True,
        )
    except Exception:  # unreadable / detector error -> do not skip the removal
        logger.debug("has_invisible_target: identify failed, defaulting to run", exc_info=True)
        return True
    # An asset edit can invalidate the C2PA binding while leaving the declared pixel
    # watermark intact. Keep the removal gate fail-safe without promoting that broken
    # claim back into the provenance verdict.
    return report.ai_from_metadata or c2pa_info_has_removal_hint(evidence.c2pa_info)

"""Image provenance: identify where an image was made and what watermarks it carries.

Aggregates every locally-readable signal into a single :class:`ProvenanceReport`:

- **C2PA Content Credentials** (issuer, claim generator, digital source type) ->
  the signing platform (OpenAI, Google, Adobe, Microsoft).
- **IPTC ``digitalSourceType``** "Made with AI" marker (Meta, X, others).
- **PNG text / EXIF generation parameters** (Stable Diffusion, ComfyUI, InvokeAI).
- **SynthID metadata proxy** -- a C2PA companion from a SynthID-using vendor
  (Google / OpenAI) implies the invisible pixel watermark.
- **Visible marks** (optional; needs cv2/numpy, no GPU): the Gemini sparkle and
  the ByteDance Doubao 豆包AI生成 / Jimeng 即梦AI text marks.

Hard limit: a stripped image (re-encoded, screenshotted, social-media upload)
loses all metadata, and the SynthID *pixel* watermark is not locally decodable
(proprietary decoder). Absence of signals is therefore reported as ``Unknown``,
never as "clean". See CLAUDE.md "SynthID detection is metadata-only".
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from remove_ai_watermarks.metadata import (
    AI_METADATA_KEYS,
    AIGC_MARKERS,
    IPTC_AI_FIELD_MARKERS,
    IPTC_AI_MARKERS,
    aigc_label,
    c2pa_cloud_manifest_in,
    c2pa_marker_in,
    exif_generator,
    get_ai_metadata,
    huggingface_job,
    iptc_ai_system,
    samsung_genai,
    scan_head,
    xai_signature,
)
from remove_ai_watermarks.noai.c2pa import cbor_text_after, extract_c2pa_info, soft_binding_vendors_in
from remove_ai_watermarks.noai.constants import (
    C2PA_AI_TOOLS,
    C2PA_AI_VENDORS,
    C2PA_IDENTITY_AI_ORGS,
    C2PA_ISSUERS,
)
from remove_ai_watermarks.watermark_registry import GEMINI_SPARKLE_TRUST_CONF

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import MarkDetection

logger = logging.getLogger(__name__)

# How much of a non-PNG container to binary-scan for the C2PA issuer.
_SCAN_BYTES = 1024 * 1024

# Visible-sparkle confidence above which the signal is trusted as provenance.
# Shared with the removal arbitration (watermark_registry.GEMINI_SPARKLE_TRUST_CONF)
# so the provenance "is there a sparkle" verdict and the removal "take the sparkle"
# decision can never drift apart -- the detect-vs-remove desync the retained-corpus
# mining surfaced (2026-06-20). On the corpus Gemini-family sparkles score >= 0.56
# while non-sparkle images top out at 0.49, so 0.5 cleanly separates them and avoids
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
    "SynthID is a metadata proxy here; the pixel watermark is not locally "
    "verifiable (proprietary decoder). Confirm via the Gemini app or openai.com/verify."
)
_OPENAI_CAVEAT = (
    "OpenAI began pairing SynthID with C2PA around 2026-05; OpenAI images from "
    "before the rollout carry C2PA without SynthID, so the SynthID verdict is 'likely'."
)
_IPTC_ONLY_CAVEAT = "The IPTC 'Made with AI' tag flags AI provenance but does not identify the specific platform."
_INVISIBLE_WM_CAVEAT = (
    "The open invisible watermark is fragile: it does not survive JPEG re-encoding "
    "or resizing, so it confirms origin only on a pristine (un-re-encoded) file."
)
_HF_JOB_CAVEAT = (
    "The hf-job-id tag marks a HuggingFace-hosted job (commonly diffusion "
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


@dataclass
class ProvenanceReport:
    """Aggregated provenance verdict for one image."""

    path: Path
    is_ai_generated: bool | None  # True / False is never asserted; None = unknown
    platform: str | None
    confidence: str  # "high" | "medium" | "none"
    # Coarse AI-origin kind from the C2PA digital-source-type, so a caller can
    # branch on full generation vs an AI-touched real photo:
    #   "generated" -- digitalSourceType trainedAlgorithmicMedia (fully AI).
    #   "enhanced"  -- compositeWithTrainedAlgorithmicMedia (real content with an
    #                  AI-composited region; scrub the AI region, keep the photo).
    #   None        -- no C2PA AI source-type (verdict, if AI, came from another
    #                  signal: IPTC, AIGC, local gen params, xAI, ...).
    ai_source_kind: str | None = None
    # True when the AI verdict rests on a metadata or embedded-invisible signal
    # (C2PA AI issuer / SynthID proxy, IPTC, AIGC, local gen params, EXIF/xAI, or
    # an open DWT-DCT / TrustMark decode) -- as opposed to a visible mark or a
    # weak medium-confidence hint (hf-job, Samsung genAIType). It is exactly the
    # set of signals an invisible/diffusion scrub targets: a visible-only or
    # no-signal image has it False. Equivalent to ``confidence == "high"``;
    # surfaced as a field so callers gate on intent, not on the string.
    ai_from_metadata: bool = False
    watermarks: list[str] = field(default_factory=list[str])
    signals: list[Signal] = field(default_factory=list["Signal"])
    caveats: list[str] = field(default_factory=list[str])
    # Contradictions between independent provenance signals (e.g. two different
    # AI vendors both claiming the image, or camera-capture credentials next to
    # AI-generation markers). Non-empty means the provenance is internally
    # inconsistent -- a strong tell of spoofed, transplanted, or laundered metadata.
    integrity_clashes: list[str] = field(default_factory=list[str])


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


def _device_platform(head: bytes) -> str | None:
    """Map a distinctive C2PA device/camera token in the manifest bytes to a platform."""
    for token, platform in _DEVICE_C2PA_PLATFORM:
        if token in head:
            return platform
    return None


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
    for token, platform in _SIGNER_C2PA_PLATFORM:
        if token in head:
            return platform
    return None


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


# Coarse origin-vendor normalization for integrity-clash detection. Two signals
# that resolve to the SAME key are consistent (a C2PA "Google (Gemini)" issuer
# and a SynthID-Google proxy, or Adobe Firefly + its Adobe TrustMark soft
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
# SynthID proxy are NOT independent -- the proxy is inferred from the same C2PA
# manifest -- so they share one source. A multi-actor manifest (a product wrapping
# another vendor's engine, e.g. Microsoft+OpenAI or Microsoft+Google; or an edit
# chain like Adobe over a Gemini original) legitimately names several vendors in
# one valid chain and must not read as spoofing. Families not listed here are each
# their own independent source (EXIF/XMP generator, IPTC AISystemUsed, AIGC, ...).
# The single C2PA-manifest source shared by the issuer attribution and the SynthID
# proxy (both inferred from the same embedded manifest). Rule 2 keys off it too:
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
            proxy, EXIF/XMP generator tag, IPTC AISystemUsed, xAI, AIGC label).
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
    # one C2PA manifest (c2pa issuer + synthid proxy) do not flag.
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
    # manifest (c2pa issuer + synthid proxy) -- a legitimate edit chain, not a
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
_VISIBLE_MARK_PLATFORM = {
    "doubao": "ByteDance Doubao (visible 豆包AI生成 mark detected)",
    "jimeng": "ByteDance Jimeng / Dreamina (visible 即梦AI mark detected)",
    "qwen": "Alibaba Tongyi Qianwen (visible 千问AI生成 mark detected)",
    "samsung": "Samsung Galaxy AI (visible 'Contenuti generati dall'AI' mark detected)",
}


def _visible_text_marks(image_path: Path, *, image: NDArray[Any] | None = None) -> list[MarkDetection]:
    """Detected visible text marks (registry ``MarkDetection`` list).

    The Gemini sparkle keeps its own ``_visible_sparkle`` path (file-level
    confidence); these two text marks reuse the registry detectors, which apply
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


def _invisible_watermark(image_path: Path) -> str | None:
    """Open invisible-watermark scheme name (SD/SDXL/FLUX) or None.

    Optional: needs the imwatermark decoder (extra ``detect``). Returns None if
    it is not installed or no known watermark decodes.
    """
    from remove_ai_watermarks.invisible_watermark import detect_invisible_watermark

    return detect_invisible_watermark(image_path)


def _trustmark(image_path: Path) -> str | None:
    """Adobe TrustMark scheme name or None.

    Optional: needs the ``trustmark`` decoder (extra ``trustmark``). Returns None
    if it is not installed or no TrustMark watermark decodes.
    """
    from remove_ai_watermarks.trustmark_detector import detect_trustmark

    return detect_trustmark(image_path)


def _collect_visible_signals(
    image_path: Path,
    signals: list[Signal],
    watermarks: list[str],
    platform: str | None,
) -> str | None:
    """Decode once, append every trusted visible-mark signal, and return platform.

    Keeping this stage separate from metadata aggregation makes the optional cv2
    boundary explicit and guarantees that all visible detectors share one decoded
    BGR array. A decode failure preserves the detectors' historical fallback/no-op
    behavior.
    """
    image: NDArray[Any] | None = None
    try:
        from remove_ai_watermarks.image_io import imread

        image = imread(image_path)
    except Exception as exc:  # cv2 missing - detectors fall back / no-op
        logger.debug("visible-mark decode unavailable: %s", exc)

    sparkle_conf = _visible_sparkle(image_path, image=image)
    if sparkle_conf is not None and sparkle_conf >= _SPARKLE_THRESHOLD:
        signals.append(Signal("visible_sparkle", f"NCC confidence {sparkle_conf:.2f}", "medium"))
        watermarks.append(f"Visible Gemini sparkle (confidence {sparkle_conf:.2f})")
        if platform is None:
            platform = "Google Gemini family (visible sparkle detected)"

    for detection in _visible_text_marks(image_path, image=image):
        signals.append(Signal(f"visible_{detection.key}", f"NCC confidence {detection.confidence:.2f}", "medium"))
        watermarks.append(f"Visible {detection.label} (confidence {detection.confidence:.2f})")
        if platform is None:
            platform = _VISIBLE_MARK_PLATFORM[detection.key]
    return platform


def identify(image_path: Path, *, check_visible: bool = True, check_invisible: bool = True) -> ProvenanceReport:
    """Identify an image's origin platform and watermark inventory.

    Args:
        image_path: Path to the image (PNG, JPEG, WebP, or ISOBMFF container).
        check_visible: Also run the visible-mark detectors (cv2) -- the Gemini
            sparkle and the Doubao/Jimeng text marks from the registry. Set
            False for a pure-metadata, dependency-light scan.
        check_invisible: Also decode open invisible watermarks (SD/SDXL/FLUX) via
            the optional imwatermark library. No-op when it is not installed.

    Returns:
        A :class:`ProvenanceReport`. ``is_ai_generated`` is True when any AI
        signal is found and None (unknown) when none is -- it is never asserted
        False, because stripped metadata leaves no local proof of a clean origin.
    """
    info = extract_c2pa_info(image_path)  # PNG-structured; {} for other formats
    meta = get_ai_metadata(image_path)  # PNG text + EXIF + C2PA fields + synthid

    # First MB covers C2PA (PNG caBX, JPEG APP11, AVIF/HEIF/JXL uuid box) and
    # IPTC markers for the non-PNG path where extract_c2pa_info returns {}.
    # scan_head also seeks out late ISOBMFF provenance boxes (manifest after a
    # large mdat in a streaming MP4) that a fixed first-MB read would miss.
    head = scan_head(image_path, _SCAN_BYTES)

    signals: list[Signal] = []
    watermarks: list[str] = []
    caveats: list[str] = []
    # One normalized origin vendor per generator-stamped signal, for integrity-
    # clash detection (see _integrity_clashes). Visible sparkle and the open
    # invisible watermark are deliberately excluded: the former is a fuzzy visual
    # score, the latter can be a by-product of our own SDXL removal pass, so
    # neither is a trustworthy "the generator stamped its identity" claim.
    ai_vendor_claims: dict[str, str] = {}
    camera_label = _device_platform(head)
    signer_label = _signer_platform(head)

    # ── C2PA Content Credentials ────────────────────────────────────
    has_c2pa = bool(info) or c2pa_marker_in(head)
    issuers = [info["issuer"]] if info.get("issuer") else _issuers_in(head)
    # Full AI generation (trainedAlgorithmicMedia) vs an AI-enhanced real photo
    # (compositeWithTrainedAlgorithmicMedia). The structured kind is parsed once in
    # noai.c2pa._populate_registry_fields (covers PNG + any container the c2pa-python
    # reader handles); fall back to a raw head scan for the non-PNG raw-blob path
    # where extract_c2pa_info returns {}. Full generation wins when both appear.
    c2pa_source_kind = info.get("ai_source_kind")
    if c2pa_source_kind is None:
        if b"trainedAlgorithmicMedia" in head:
            c2pa_source_kind = "generated"
        elif b"compositeWithTrainedAlgorithmicMedia" in head:
            c2pa_source_kind = "enhanced"
    # An identity-AI issuer (a pure-generator brand like Dreamina) asserts AI even
    # without a digitalSourceType -- some ByteDance/Dreamina manifests ship no
    # trainedAlgorithmicMedia, so the registered generator name is the only signal.
    # Restricted to the ``asserts_ai`` vendors (distinctive brand strings), so it
    # does not reopen the incidental-mention problem the common-word issuers have.
    issuer_blob = " ".join(issuers)
    c2pa_identity_ai = has_c2pa and any(org in issuer_blob for org in C2PA_IDENTITY_AI_ORGS)
    c2pa_is_ai = c2pa_source_kind is not None or c2pa_identity_ai
    # Generator string (for the signal detail): structured for PNG, CBOR-scanned
    # for other containers. Best-effort -- some manifests key it as
    # `claim_generator_info` (Pixel), so this can be None even when a device is
    # identified by `_device_platform`.
    generator = (
        info.get("claim_generator")
        or cbor_text_after(head, b"claim_generator")
        or (", ".join(tools) if (tools := _ai_tools_in(head)) else None)
    )
    # Platform: a distinctive device/camera token in the manifest wins (it is the
    # signer/producer), then an editing-app/AI-device signer (Samsung Galaxy,
    # ASUS Gallery), with the issuer byte-scan only as fallback. The issuer scan
    # alone mis-attributed real samples (Leica->Truepic timestamp authority,
    # Nikon->Adobe namespace, Pixel->Google Gemini) -- the token scans fix that.
    platform = (camera_label or signer_label or _attribute_platform(issuers, is_ai=c2pa_is_ai)) if has_c2pa else None
    if has_c2pa:
        detail = ", ".join(filter(None, [", ".join(issuers), generator, info.get("source_type")]))
        signals.append(Signal("c2pa", detail or "C2PA manifest present", "high"))
        watermarks.append(f"C2PA Content Credentials ({', '.join(issuers) or 'unknown signer'})")
        # Record the AI-origin vendor for clash detection only when the source is
        # actually AI -- classify the issuer attribution / generator, NOT the
        # resolved `platform` (which may be a camera device token whose label,
        # e.g. "Google Pixel", would mis-normalize to an AI vendor).
        if c2pa_is_ai and (v := (_vendor_of(_attribute_platform(issuers, is_ai=True)) or _vendor_of(generator))):
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

    # ── SynthID metadata proxy ──────────────────────────────────────
    # get_ai_metadata already sets synthid_watermark for both PNG (caBX parser)
    # and non-PNG (its own synthid_source fallback), so no extra scan is needed.
    synthid = meta.get("synthid_watermark")
    if synthid:
        watermarks.append(f"SynthID watermark, inferred from C2PA metadata ({synthid})")
        caveats.append(_SYNTHID_CAVEAT)
        if _vendor_of(synthid) == "OpenAI":
            caveats.append(_OPENAI_CAVEAT)
        if v := _vendor_of(synthid):
            ai_vendor_claims["synthid"] = v

    # ── C2PA soft-binding: a named forensic/third-party watermark vendor ─
    # (Adobe TrustMark, Digimarc, Imatag, ...). Present in the manifest even when
    # the watermark itself can't be decoded; names whose watermark stamped the pixels.
    soft_binding = meta.get("soft_binding") or (", ".join(v) if (v := soft_binding_vendors_in(head)) else None)
    if soft_binding:
        signals.append(Signal("soft_binding", f"C2PA soft binding: {soft_binding}", "high"))
        watermarks.append(f"Forensic watermark soft binding ({soft_binding})")

    # ── IPTC "Made with AI" (Meta etc.), only meaningful without C2PA ─
    iptc = any(m in head for m in IPTC_AI_MARKERS)
    if iptc and not has_c2pa:
        signals.append(Signal("iptc", "digitalSourceType (Made with AI)", "high"))
        watermarks.append("IPTC digitalSourceType (Made with AI)")
        caveats.append(_IPTC_ONLY_CAVEAT)
        if platform is None:
            platform = "Made-with-AI tag (e.g. Meta AI); platform not specified"

    # ── IPTC 2025.1 AI-disclosure fields (Iptc4xmpExt:AISystemUsed etc.) ─
    iptc_ai = any(m in head for m in IPTC_AI_FIELD_MARKERS)
    if iptc_ai:
        system = iptc_ai_system(image_path)
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
    aigc_data = aigc_label(image_path)
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
    if generator_tag := exif_generator(image_path):
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
    if xai_signature(image_path):
        signals.append(Signal("xai_signature", "EXIF Signature blob + UUID Artist", "high"))
        watermarks.append("xAI/Grok EXIF signature")
        if platform is None:
            platform = "xAI (Grok / Aurora)"
        ai_vendor_claims["xai"] = "xAI"

    # ── HuggingFace-hosted job marker (hf-job-id PNG text chunk) ─────
    # Marks the hosting job, not a model -- medium confidence (commonly diffusion
    # output). Like the visible sparkle, it lifts an otherwise-Unknown verdict to
    # a tentative AI, but never overrides a high-confidence metadata signal.
    hf_job = huggingface_job(image_path)
    if hf_job:
        signals.append(Signal("hf_job", f"HuggingFace job {hf_job}", "medium"))
        watermarks.append("HuggingFace-hosted job (hf-job-id)")
        caveats.append(_HF_JOB_CAVEAT)
        if platform is None:
            platform = "HuggingFace-hosted job (model not identified)"

    # ── Samsung Galaxy AI editing marker (genAIType) ─────────────────
    # Galaxy AI tools stamp a proprietary genAIType in PhotoEditor_Re_Edit_Data.
    # Medium confidence: it co-occurs with the C2PA trainedAlgorithmicMedia type
    # on Galaxy files that record one, and is the SOLE AI marker on a Galaxy S24
    # sample that omits the source type -- so it lifts an otherwise-Unknown
    # verdict, but the field is undocumented, so it never overrides a high-
    # confidence signal. The platform is usually already "Samsung Galaxy" via the
    # signer-token scan; the fallback covers a future file without the cert org.
    samsung_genai_type = samsung_genai(image_path)
    if samsung_genai_type is not None:
        signals.append(Signal("samsung_genai", f"Samsung genAIType={samsung_genai_type}", "medium"))
        watermarks.append("Samsung Galaxy AI editing marker (genAIType)")
        caveats.append(_SAMSUNG_GENAI_CAVEAT)
        if platform is None:
            platform = "Samsung Galaxy (Galaxy AI editing)"

    # ── Open invisible watermark (SD / SDXL / FLUX, dwtDct) ──────────
    # Public decoder, no key -- a definitive embedded signal on pristine files.
    if check_invisible and (scheme := _invisible_watermark(image_path)) is not None:
        signals.append(Signal("invisible_watermark", scheme, "high"))
        watermarks.append(f"Open invisible watermark: {scheme}")
        caveats.append(_INVISIBLE_WM_CAVEAT)
        if platform is None:
            platform = f"{scheme} (open DWT-DCT watermark)"

    # ── Adobe TrustMark invisible watermark (open decoder, no key) ───
    # The watermark behind Adobe Durable Content Credentials. Decoded locally,
    # but it binds provenance for human-authored content too, so it enriches the
    # watermark inventory without by itself asserting AI origin.
    if check_invisible and (tm_scheme := _trustmark(image_path)) is not None:
        signals.append(Signal("trustmark", tm_scheme, "high"))
        watermarks.append(f"Adobe TrustMark invisible watermark ({tm_scheme})")
        if platform is None:
            platform = "Adobe (TrustMark / Content Credentials)"

    # ── Verdict so far (metadata + embedded watermark) ──────────────
    invisible_wm = any(s.name == "invisible_watermark" for s in signals)
    exif_gen = any(s.name == "exif_generator" for s in signals)
    xai_sig = any(s.name == "xai_signature" for s in signals)
    ai_from_metadata = bool(
        (has_c2pa and (c2pa_is_ai or synthid))
        or iptc
        or iptc_ai
        or aigc
        or local_keys
        or invisible_wm
        or exif_gen
        or xai_sig
    )

    if check_visible:
        platform = _collect_visible_signals(image_path, signals, watermarks, platform)

    visible_only = any(s.name.startswith("visible_") for s in signals) and not ai_from_metadata
    hf_only = bool(hf_job) and not ai_from_metadata
    samsung_only = samsung_genai_type is not None and not ai_from_metadata

    if ai_from_metadata:
        is_ai: bool | None = True
        confidence = "high"
    elif visible_only or hf_only or samsung_only:
        is_ai = True
        confidence = "medium"
    else:
        is_ai = None
        confidence = "none"

    # ── Integrity clashes: contradictions between independent signals ─
    clashes = _integrity_clashes(ai_vendor_claims, camera_label, camera_has_ai_marker=bool(ai_vendor_claims))

    caveats.append(_STRIP_CAVEAT)
    # De-duplicate while preserving order.
    caveats = list(dict.fromkeys(caveats))

    return ProvenanceReport(
        path=image_path,
        is_ai_generated=is_ai,
        platform=platform,
        confidence=confidence,
        # Only meaningful when the AI verdict actually came from the C2PA source
        # type; a non-C2PA AI signal (IPTC/AIGC/local gen) leaves it None.
        ai_source_kind=c2pa_source_kind if (is_ai and has_c2pa) else None,
        ai_from_metadata=ai_from_metadata,
        watermarks=watermarks,
        signals=signals,
        caveats=caveats,
        integrity_clashes=clashes,
    )


def has_invisible_target(image_path: Path) -> bool:
    """True when a locally-detectable invisible/metadata AI signal is present.

    The decision gate for the diffusion scrub (``invisible`` / ``all`` / ``batch``):
    regenerating pixels removes an invisible watermark (SynthID, open DWT-DCT,
    TrustMark) but degrades a real photo, so it must not run when there is nothing
    to remove. Runs :func:`identify` with ``check_visible=False`` -- a visible mark
    is handled by the separate visible pass and is NOT a diffusion target -- and
    ``check_invisible=True`` so an open watermark counts. Returns
    ``report.ai_from_metadata`` (C2PA AI issuer / SynthID proxy, IPTC, AIGC, local
    gen params, EXIF/xAI, open DWT-DCT / TrustMark).

    IMPORTANT -- this cannot prove a pixel SynthID is absent: SynthID is detectable
    only through its C2PA proxy, so a metadata-stripped AI image reads as no signal
    here. A False therefore means "no locally-detectable invisible target", not
    "clean". Callers must NOT present a skip as a finished clean result.

    Fail-safe: any error resolves to True so the removal still runs -- leaving a
    watermark on a paid removal is worse than over-regenerating a clean image.
    """
    try:
        report = identify(image_path, check_visible=False, check_invisible=True)
    except Exception:  # unreadable / detector error -> do not skip the removal
        logger.debug("has_invisible_target: identify failed, defaulting to run", exc_info=True)
        return True
    return report.ai_from_metadata

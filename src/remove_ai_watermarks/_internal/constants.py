"""Registries shared by metadata extraction and provenance classification."""

from __future__ import annotations

from dataclasses import dataclass

from remove_ai_watermarks._internal._generated_c2pa_soft_bindings import C2PA_SOFT_BINDING_ROWS


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(value.split("|"))


SUPPORTED_FORMATS = frozenset(_tokens(".png|.jpg|.jpeg|.webp|.heic|.heif|.avif"))
AI_METADATA_KEYS = _tokens(
    "parameters|postprocessing|extras|workflow|prompt|Dream|SD:mode|StableDiffusionVersion|"
    "generation_time|Model|Model hash|Seed"
)
AI_KEYWORDS = _tokens(
    "prompt|negative_prompt|sampler|cfg_scale|lora|diffusion|comfy|midjourney|dall-e|dalle|imagen|firefly|c2pa|chatgpt|gpt-4|sora|openai|truepic|stable_diffusion|invokeai"
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
C2PA_CHUNK_TYPE = b"caBX"
PNG_METADATA_CHUNKS = frozenset({b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"iCCP"})
RIFF_METADATA_CHUNKS = frozenset({b"EXIF", b"XMP ", b"ICCP", b"C2PA"})
RIFF_CODED_IMAGE_CHUNKS = frozenset({b"VP8 ", b"VP8L", b"ALPH", b"ANMF"})
C2PA_SIGNATURES = tuple(
    token.encode() for token in _tokens("c2pa|C2PA|jumb|jumd|JUMBF|jumbf|cbor|contentcreds|digid|assertions|manifest")
)


@dataclass(frozen=True, slots=True)
class C2paAiVendor:
    """One C2PA identity and its AI-product or neutral-signer attribution."""

    issuer: bytes
    org: str
    platform: str | None
    needle: str | None
    synthid: bool = False
    asserts_ai: bool = False
    synthid_requires_watermark_action: bool = False
    signer_platform: str | None = None
    raw_signer_platforms: tuple[tuple[bytes, str], ...] = ()


def _vendor(
    issuer: bytes | str,
    org: str,
    platform: str | None,
    needle: str | None,
    *,
    synthid: bool = False,
    asserts_ai: bool = False,
    synthid_requires_watermark_action: bool = False,
    signer_platform: str | None = None,
    raw_signer_platforms: tuple[tuple[bytes, str], ...] = (),
) -> C2paAiVendor:
    token = issuer.encode() if isinstance(issuer, str) else issuer
    return C2paAiVendor(
        token,
        org,
        platform,
        needle,
        synthid=synthid,
        asserts_ai=asserts_ai,
        synthid_requires_watermark_action=synthid_requires_watermark_action,
        signer_platform=signer_platform,
        raw_signer_platforms=raw_signer_platforms,
    )


# Order is product priority when a manifest mentions more than one organization.
C2PA_AI_VENDORS: tuple[C2paAiVendor, ...] = (
    _vendor(b"Microsoft", "Microsoft", "Microsoft (Copilot / Designer)", "Microsoft"),
    _vendor(b"Adobe", "Adobe", "Adobe Firefly", "Adobe"),
    _vendor(
        b"OpenAI",
        "OpenAI",
        "OpenAI (ChatGPT / GPT Image / DALL·E / Sora)",
        "OpenAI",
        synthid=True,
        synthid_requires_watermark_action=True,
    ),
    _vendor(b"Google", "Google LLC", "Google (Gemini / Imagen)", "Google", synthid=True),
    _vendor(b"Stability AI", "Stability AI", "Stability AI (Stable Image / DreamStudio)", "Stability AI"),
    _vendor(b"Black Forest Labs", "Black Forest Labs", "Black Forest Labs (FLUX)", "Black Forest Labs"),
    _vendor(
        b"volcengine",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    _vendor(
        "北京火山引擎科技有限公司",
        "ByteDance (Volcano Engine)",
        "ByteDance Volcano Engine",
        "Volcano Engine",
    ),
    _vendor(b"Byteplus", "BytePlus (ByteDance)", "BytePlus (ByteDance)", "BytePlus"),
    _vendor(
        b"Dreamina",
        "ByteDance (Dreamina)",
        "ByteDance Dreamina",
        "Dreamina",
        asserts_ai=True,
    ),
    _vendor(b"Canva", "Canva", "Canva (Magic Media)", "Canva"),
    _vendor(b"Eleven Labs", "ElevenLabs", "ElevenLabs", "ElevenLabs"),
    _vendor(b"fal-ai", "fal.ai", "fal.ai", "fal.ai", asserts_ai=True),
    _vendor(b"Bria", "Bria Artificial Intelligence", "Bria AI", "Bria", asserts_ai=True),
    # Ideogram signs its downloads' Content Credentials with "Ideogram, Inc"; the
    # issuer token is the org prefix (same substring-match class as "Bria" in
    # "Bria Artificial Intelligence"). Found as an unmapped signer on 4 corpus
    # uploads 2026-08-08 that identify reported as unknown-signer C2PA.
    _vendor(b"Ideogram", "Ideogram", "Ideogram", "Ideogram", asserts_ai=True),
    _vendor(b"xAI Grok Imagine", "xAI Grok Imagine", "xAI Grok Imagine", "xAI", asserts_ai=True),
    _vendor(b"Producer.ai", "Producer.ai", "Producer.ai", "Producer.ai", asserts_ai=True),
    # This signature names only SPRING's legal entity, not a product. Preserve
    # that signer identity without guessing which product produced a file.
    _vendor(b"SPRING (SG) PTE. LTD.", "SPRING (SG) PTE. LTD.", "SPRING (SG) PTE. LTD.", "SPRING"),
    # These companies also sign ordinary edits and publications. Register the
    # signer identity for provenance display, but provide no AI platform and do
    # not let the identity assert an AI verdict.
    _vendor(
        b"TikTok Inc.",
        "TikTok",
        None,
        None,
        signer_platform="TikTok (C2PA signer)",
    ),
    _vendor(
        b"Bytedance Pte",
        "ByteDance",
        None,
        None,
        signer_platform="ByteDance (C2PA signer)",
        raw_signer_platforms=(
            (b"Bytedance Pte. Ltd.", "ByteDance (C2PA signer)"),
            (b"CapCut/", "CapCut (C2PA signer)"),
            (b"capcut c2pa-rs", "CapCut (C2PA signer)"),
        ),
    ),
    _vendor(
        b"Anthropic Claude Content Signing",
        "Anthropic Claude",
        None,
        None,
        signer_platform="Anthropic Claude (C2PA signer)",
    ),
    _vendor(
        b"Samsung Galaxy",
        "Samsung Galaxy",
        None,
        None,
        signer_platform="Samsung Galaxy (C2PA)",
    ),
    _vendor(
        b"com.asus.gallery",
        "ASUS Gallery",
        None,
        None,
        signer_platform="ASUS Gallery (C2PA signer)",
    ),
    # Certificate common name on Google Photos edits (Android and iOS). The
    # manifest also names Google LLC, so an AI edit establishes Google SynthID
    # (see ``SYNTHID_EDIT_SIGNERS``) and its platform label lives in ``identify``; an edited
    # Gemini or OpenAI generation keeps its generator.
    _vendor(b"Google Photos", "Google Photos", None, None),
    # Runway signs only its own models (Gen-4 image, Gen-4.5 video, measured
    # 2026-09-24: issuer "RUNWAY AI, INC.", software agent "Runway Image/Video
    # Generation"); a third-party model run inside Runway keeps its vendor's own
    # manifest (a Nano Banana 2 image came back with Google's). The C2PA
    # conformance list spells the organization "Runway AI, Inc".
    _vendor(b"RUNWAY AI, INC.", "Runway", "Runway", "Runway"),
    _vendor(b"Runway AI, Inc", "Runway", "Runway", "Runway"),
    _vendor(b"Truepic", "Truepic", None, None),
)

C2PA_ISSUERS = {vendor.issuer: vendor.org for vendor in C2PA_AI_VENDORS}
# Google signers that re-encode media rather than generate it, so their Google
# LLC identity alone says nothing about SynthID. YouTube re-signs every upload
# with "YouTube Video Processing Services" (opened and transcoded, measured
# 2026-09-24): an xAI Grok video came back reading as Google SynthID. A manifest
# from one of these signers establishes SynthID only when the chain records a
# SynthID action, as a Gemini ingredient does. Google Photos is deliberately
# absent: its manifests record no SynthID action, yet Google's checker found
# SynthID on all four Photos AI edits tested (Ask, eraser and two other edits,
# 2026-09-25), although Google documents the mark only for Reimagine.
SYNTHID_EDIT_SIGNERS: tuple[bytes, ...] = (b"YouTube",)
C2PA_IDENTITY_AI_ORGS = frozenset(vendor.org for vendor in C2PA_AI_VENDORS if vendor.asserts_ai)
C2PA_SIGNER_PLATFORM_BY_ORG = {
    vendor.org: vendor.signer_platform for vendor in C2PA_AI_VENDORS if vendor.signer_platform is not None
}
C2PA_SIGNER_PLATFORMS = tuple(
    pair
    for vendor in C2PA_AI_VENDORS
    if vendor.signer_platform is not None
    for pair in (vendor.raw_signer_platforms or ((vendor.issuer, vendor.signer_platform),))
)

# Product-specific claim generators can sign through a different upstream issuer.
# Keep this attribution beside the issuer registry so every C2PA consumer has one
# canonical source rather than maintaining a derived product map in identify.py.
C2PA_CLAIM_GENERATOR_PLATFORMS: tuple[tuple[str, str], ...] = (
    ("grok imagine", "xAI Grok Imagine"),
    ("suno", "Suno"),
    ("adobe_firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("dreamina", "ByteDance Dreamina"),
    ("higgsfield ai", "Higgsfield AI"),
    ("recraft.ai", "Recraft"),
    ("topaz labs image api", "Topaz Labs"),
    ("tiktok ad creative toolbox", "TikTok Ad Creative Toolbox"),
    ("fastvid", "FastVid"),
    ("capcut", "CapCut (C2PA signer)"),
)

C2PA_AI_TOOLS = {
    token.encode(): label
    for token, label in (
        ("GPT-4o", "GPT-4o"),
        ("ChatGPT", "ChatGPT"),
        ("Sora", "Sora"),
        ("DALL-E", "DALL·E"),
        ("DALL", "DALL·E"),
        ("Imagen", "Imagen"),
        ("Firefly", "Firefly"),
        ("Dreamina", "Dreamina"),
    )
}


@dataclass(frozen=True, slots=True)
class C2paSoftBindingAlgorithm:
    """One normalized entry from the official C2PA soft-binding registry."""

    identifier: int
    algorithm: str
    kind: str
    decoded_media_types: tuple[str, ...]
    encoded_media_types: tuple[str, ...]
    display_label: str
    date_entered: str
    resolution_apis: tuple[str, ...]
    deprecated: bool


# These compact historical labels are the public display contract. The generated
# snapshot supplies every exact registered algorithm and its official-description
# fallback, while these prefixes keep existing output concise and stable.
_C2PA_SOFT_BINDING_LABEL_OVERRIDES = {
    b"com.adobe.trustmark": "Adobe TrustMark",
    b"com.adobe.icn": "Adobe Image Comparator Network",
    b"com.digimarc": "Digimarc Validate",
    b"com.imatag.lamark": "Imatag (Lamark)",
    b"ai.steg": "Steg.AI",
    b"com.microsoft.invismark": "Microsoft InvisMark",
    b"com.microsoft.wavmark": "Microsoft WavMark",
    b"com.verimatrix": "Verimatrix",
    b"com.nagra.nexguard": "NAGRA NexGuard",
    b"com.aiwatermark.pixelseal": "AIWatermark PixelSeal",
    b"com.aiwatermark.videoseal": "AIWatermark VideoSeal",
    b"com.aiwatermark.audioseal": "AIWatermark AudioSeal",
    b"ai.trufo": "Trufo PawPrint",
    b"app.overlai": "Overlai",
    b"com.markany": "MarkAny",
    b"com.mentaport": "Mentaport",
    b"es.lumatrace": "LumaTrace",
    b"ai.verda": "VerdaAI",
    b"ai.contentlens": "ContentLens",
    b"io.iscc": "ISCC (content code)",
}

C2PA_SOFT_BINDING_REGISTRY = tuple(C2paSoftBindingAlgorithm(*row) for row in C2PA_SOFT_BINDING_ROWS)


def _c2pa_soft_binding_label(entry: C2paSoftBindingAlgorithm) -> str:
    encoded_algorithm = entry.algorithm.encode()
    return next(
        (label for prefix, label in _C2PA_SOFT_BINDING_LABEL_OVERRIDES.items() if encoded_algorithm.startswith(prefix)),
        entry.display_label,
    )


C2PA_SOFT_BINDINGS = {entry.algorithm.encode(): _c2pa_soft_binding_label(entry) for entry in C2PA_SOFT_BINDING_REGISTRY}

AI_GENERATOR_TOKENS = frozenset(
    {
        "firefly",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "stablediffusion",
        "comfyui",
        "automatic1111",
        "invokeai",
        "imagen",
        "gpt-image",
        "nano banana",
        "nightcafe",
        "ideogram",
        "leonardo",
        "flux",
        "dreamstudio",
        "novelai",
        "reve.com",
        # Luma AI stamps PNG tEXt Source="Luma AI" / Comment="Generated by
        # Luma AI's Uni-1 model (https://lumalabs.ai)"; the space-bearing token
        # avoids matching incidental "luma" runs (luma/chroma key names etc.).
        "luma ai",
        "lumalabs",
        "aphrodite ai",
        "apple photos clean up",
        "apple photos generative edit",
        "apple image playground",
        "fal-ai",
    }
)

_C2PA_ACTION_NAMES = _tokens("created|converted|edited|filtered|cropped|resized|opened|placed|watermarked.unbound")
C2PA_ACTIONS = {f"c2pa.{action}".encode(): action for action in _C2PA_ACTION_NAMES}

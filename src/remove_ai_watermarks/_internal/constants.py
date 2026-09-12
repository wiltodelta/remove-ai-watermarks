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
    """One issuer signature and its normalized product attribution."""

    issuer: bytes
    org: str
    platform: str | None
    needle: str | None
    synthid: bool = False
    asserts_ai: bool = False
    synthid_requires_watermark_action: bool = False


def _vendor(
    issuer: bytes | str,
    org: str,
    platform: str | None,
    needle: str | None,
    *,
    synthid: bool = False,
    asserts_ai: bool = False,
    synthid_requires_watermark_action: bool = False,
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
    _vendor(b"Truepic", "Truepic", None, None),
)

C2PA_ISSUERS = {vendor.issuer: vendor.org for vendor in C2PA_AI_VENDORS}
C2PA_IDENTITY_AI_ORGS = frozenset(vendor.org for vendor in C2PA_AI_VENDORS if vendor.asserts_ai)

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
        "fal-ai",
    }
)

_C2PA_ACTION_NAMES = _tokens("created|converted|edited|filtered|cropped|resized|opened|placed|watermarked.unbound")
C2PA_ACTIONS = {f"c2pa.{action}".encode(): action for action in _C2PA_ACTION_NAMES}

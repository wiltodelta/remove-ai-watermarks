"""Shared constants for AI metadata detection, C2PA parsing, and format support.

All modules reference these constants rather than hard-coding values,
so adding a new AI tool or metadata key requires updating only this file.
"""

# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

# AI-generated image metadata keys (Stable Diffusion, ComfyUI, Midjourney, etc.)
AI_METADATA_KEYS = [
    "parameters",  # Stable Diffusion WebUI (AUTOMATIC1111, Vladmandic)
    "postprocessing",  # SD WebUI post-processing info
    "extras",  # SD WebUI extras
    "workflow",  # ComfyUI workflow JSON
    "prompt",  # Some AI tools
    "Dream",  # DreamStudio
    "SD:mode",  # Stability AI
    "StableDiffusionVersion",  # SD version info
    "generation_time",  # Generation time info
    "Model",  # Model name
    "Model hash",  # Model hash
    "Seed",  # Seed value
]

# Standard PNG metadata keys
PNG_METADATA_KEYS = [
    "Author",
    "Title",
    "Description",
    "Copyright",
    "Creation Time",
    "Software",
    "Disclaimer",
    "Warning",
    "Source",
    "Comment",
]

# AI-related keywords for detection
AI_KEYWORDS = [
    "prompt",
    "negative_prompt",
    "sampler",
    "cfg_scale",
    "lora",
    "diffusion",
    "comfy",
    "midjourney",
    "dall-e",
    "dalle",
    "imagen",
    "firefly",
    "c2pa",
    "chatgpt",
    "gpt-4",
    "sora",
    "openai",
    "truepic",
    "stable_diffusion",
    "invokeai",
]

# C2PA (Coalition for Content Provenance and Authenticity) constants
# Used by Google Imagen, Adobe Firefly, Microsoft Designer, OpenAI, etc.
C2PA_CHUNK_TYPE = b"caBX"  # JUMBF container chunk type for C2PA
C2PA_SIGNATURES = [
    b"c2pa",
    b"C2PA",
    b"jumb",
    b"jumd",
    b"JUMBF",
    b"jumbf",
    b"cbor",
    b"contentcreds",
    b"digid",
    b"assertions",
    b"manifest",
]

# C2PA known issuers
C2PA_ISSUERS = {
    b"Google": "Google LLC",
    b"Adobe": "Adobe",
    b"Microsoft": "Microsoft",
    b"OpenAI": "OpenAI",
    b"Truepic": "Truepic",
    # Stability AI signs C2PA as "Stability AI" (cert org "Stability AI Ltd").
    # Verified on a live Brand Studio (DreamStudio successor) output, 2026-05-24.
    b"Stability AI": "Stability AI",
    # Black Forest Labs (FLUX) API output: claim_generator_info "Black Forest
    # Labs API" + a c2pa.ai_generated_content assertion + trainedAlgorithmicMedia.
    # Verified on a real signed FLUX JPEG, 2026-05-29.
    b"Black Forest Labs": "Black Forest Labs",
    # ByteDance's Volcano Engine (Volcengine) signs its AI image output with a
    # cert from certificate_center@volcengine.com -- the platform behind Doubao /
    # Jimeng. Verified on two real signed JPEGs, 2026-05-29.
    b"volcengine": "ByteDance (Volcano Engine)",
}

# C2PA issuers whose signed outputs also carry an invisible SynthID pixel
# watermark -- a metadata proxy for "SynthID is in the pixels":
#   - Google (Imagen/Gemini): embeds SynthID, long-standing (DeepMind docs).
#   - OpenAI (ChatGPT/Codex/API): pairs SynthID with C2PA since ~2026-05-20.
#     Confirmed by OpenAI's Help Center ("C2PA and SynthID in OpenAI-generated
#     images", updated 2026-05-21): "Images generated with ChatGPT, Codex, and
#     our API include both C2PA metadata and SynthID watermarks." OpenAI also
#     notes a signal may be absent if "the image was created before these
#     signals were available" -- so OpenAI images from BEFORE the rollout carry
#     C2PA WITHOUT SynthID (e.g. data/samples/openai-images-2/amur-leopard.png,
#     C2PA timestamp 2026-04-22). For OpenAI the proxy is therefore "likely",
#     not certain; the verdict string is hedged accordingly. OpenAI's own oracle
#     is openai.com/verify (Google's is the Gemini app "Verify with SynthID").
# The issuer byte ("OpenAI"/"Google") is verified locally against data/samples;
# the SynthID pairing is documented behavior (Google: DeepMind; OpenAI: above).
# Adobe Firefly and Microsoft Designer sign C2PA but do NOT use SynthID, so a
# C2PA manifest alone is not a SynthID signal -- the issuer is. The pixel
# watermark is not locally detectable (proprietary decoder); the C2PA companion
# is the proxy, and only while the manifest is intact.
SYNTHID_C2PA_ISSUERS = frozenset({b"Google", b"OpenAI"})

# C2PA known AI tools
C2PA_AI_TOOLS = {
    b"GPT-4o": "GPT-4o",
    b"ChatGPT": "ChatGPT",
    b"Sora": "Sora",
    b"DALL-E": "DALL-E",
    b"DALL": "DALL-E",
    b"Imagen": "Imagen",
    b"Firefly": "Firefly",
}

# C2PA ``c2pa.soft-binding`` algorithm identifiers -> the forensic-watermark
# vendor that stamped the pixels. The manifest's ``alg`` field names the
# watermark scheme even when the watermark itself cannot be decoded locally, so
# a byte-scan for these (keyed on a distinctive prefix to catch all variants)
# tells us a third-party forensic watermark is present and whose. Verified
# against the official C2PA registry (github.com/c2pa-org/softbinding-algorithm-list).
# Adobe TrustMark is additionally decodable locally (see ``trustmark_detector``);
# the rest (Digimarc, Imatag, Steg.AI, etc.) are proprietary oracle-only decoders.
C2PA_SOFT_BINDINGS = {
    b"com.adobe.trustmark": "Adobe TrustMark",
    b"com.adobe.icn": "Adobe (content fingerprint)",
    b"com.digimarc": "Digimarc",
    b"com.imatag.lamark": "Imatag (Lamark)",
    b"ai.steg": "Steg.AI",
    b"com.microsoft.invismark": "Microsoft InvisMark",
    b"com.microsoft.wavmark": "Microsoft WavMark",
    b"com.verimatrix": "Verimatrix",
    b"com.nagra.nexguard": "NAGRA NexGuard",
    b"com.aiwatermark": "AIWatermark (Meta PixelSeal)",
    b"ai.trufo": "Trufo",
    b"app.overlai": "Overlai",
    b"com.markany": "MarkAny",
    b"com.mentaport": "Mentaport",
    b"es.lumatrace": "LumaTrace",
    b"ai.verda": "VerdaAI",
    b"ai.contentlens": "ContentLens",
    b"io.iscc": "ISCC (content code)",
}

# Lowercased substrings that mark an AI generator when found in an EXIF
# ``Software`` / XMP ``CreatorTool`` value. Conservative on purpose: plain
# editors like "Adobe Photoshop" or "GIMP" must NOT match (no AI token), so only
# generator names land here. Add new generators here, not inline.
AI_GENERATOR_TOKENS: frozenset[str] = frozenset(
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
        "nightcafe",
        "ideogram",
        "leonardo",
        "flux",
        "dreamstudio",
    }
)

# C2PA action types
C2PA_ACTIONS = {
    b"c2pa.created": "created",
    b"c2pa.converted": "converted",
    b"c2pa.edited": "edited",
    b"c2pa.filtered": "filtered",
    b"c2pa.cropped": "cropped",
    b"c2pa.resized": "resized",
    b"c2pa.opened": "opened",
    b"c2pa.placed": "placed",
}

# PNG signature
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

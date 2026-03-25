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
}

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

# C2PA action types
C2PA_ACTIONS = {
    b"c2pa.created": "created",
    b"c2pa.converted": "converted",
    b"c2pa.edited": "edited",
    b"c2pa.filtered": "filtered",
    b"c2pa.cropped": "cropped",
    b"c2pa.resized": "resized",
}

# PNG signature
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

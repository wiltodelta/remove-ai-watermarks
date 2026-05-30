"""Watermark removal model profiles, strength presets, and profile detection.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CTRLREGEN_MODEL_ID = "yepengliu/ctrlregen"

# Denoising-strength presets for the SDXL img2img scrub.
LOW_STRENGTH = 0.04
MEDIUM_STRENGTH = 0.35
HIGH_STRENGTH = 0.7

# Default strength for invisible-watermark removal. Raised from the old 0.04/0.05
# because that no longer removes the CURRENT Google SynthID (Nano Banana / Gemini
# 3): verified 2026-05-30 via the Gemini "Verify with SynthID" oracle on a real
# image -- 0.05 still detected, 0.10 not detected (OpenAI's SynthID was already
# cleared at 0.05). 0.10 keeps the visible change modest while removing both.
# CAVEAT: confirmed on n=1 Google + n=1 OpenAI image; broad oracle validation
# across the corpus is pending (different images may need a different strength).
# At this higher strength small text deforms more -- which is exactly why text
# protection (`_run_region_hires`) runs by default.
DEFAULT_STRENGTH = 0.10

_HIGH_PERTURBATION = ("stegasamp", "stegastamp", "treering", "ringid")
_LOW_PERTURBATION = ("stablesignature", "dwtectsvd", "rivagan", "ssl", "hidden")


def get_model_id_for_profile(profile: str) -> str:
    """Map CLI model profile names to concrete Hugging Face model IDs."""
    normalized = profile.strip().lower()
    if normalized == "default":
        return DEFAULT_MODEL_ID
    if normalized == "ctrlregen":
        return CTRLREGEN_MODEL_ID
    raise ValueError(f"Unknown model profile '{profile}'. Use one of: default, ctrlregen.")


def detect_model_profile(model_id: str) -> str:
    """Infer model profile from model identifier."""
    if "ctrlregen" in model_id.lower():
        return "ctrlregen"
    return "default"


def get_recommended_strength(watermark_type: str) -> float:
    """Get recommended strength for different watermark types.

    Args:
        watermark_type: Type of watermark. One of: 'low', 'medium', 'high',
                        or specific names like 'stegastamp', 'treering', etc.

    Returns:
        Recommended strength value.
    """
    wt = watermark_type.lower()
    if any(name in wt for name in _HIGH_PERTURBATION):
        return HIGH_STRENGTH
    if any(name in wt for name in _LOW_PERTURBATION):
        return LOW_STRENGTH
    return MEDIUM_STRENGTH
